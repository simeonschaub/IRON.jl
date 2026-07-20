# The tiled-*reduction* schedule behind `@iron`'s `for` form -- the shape `@iron`'s
# call form cannot infer. A GEMM is the archetype: an output tile is held across an
# inner loop that streams the input operands and accumulates into it, and each
# operand's tile is addressed by a different mix of the loop variables. That schedule
# is not visible in the kernels, so it is written as a `for` loop -- the header is the
# output-tile iteration (the space axes) and the body declares the operands and the
# reduction:
#
#     @iron stack_size = 3328 for mi in 1:div(M, m), nj in 1:div(N, n)   # output-tile (space) loop
#         @init gemm_zero!(C)                                    # zero the accumulator, once per tile
#         @reduce for kk in 1:div(K, k)                          # the accumulation (reduce) loop
#             gemm_acc!(In(A)[mi, kk], In(B)[kk, nj], Out(C)[mi, nj])  # step: C += A*B, each operand's
#         end                                                    #   [axes] indexing its tile in buffer order
#     end
#
# The generated design is the classic tiled-GEMM loop nest: the core acquires and
# initialises the output tiles in the outer (space) loop and acquires the inputs and
# reduces into them in the inner (reduce) loop, while the host DMA streams, per output
# tile, the reduction's input tiles and then drains the output. A tile shape is not
# annotated -- it follows from the buffer shape and the extents of the axes indexing
# it: buffer dimension `d` of extent `D`, indexed by an axis of extent `E`, gives a
# tile extent `D ÷ E`.
#
# `@cores <axis>` spreads a space axis across the compute-core array: that axis becomes
# *spatial* (one output-tile position per core, run concurrently) while the remaining
# space axes stay *temporal* (each core iterates them itself). One core reduces its own
# output tiles exactly as the single-core design does; the host DMA feeds every core per
# temporal step so they run in parallel:
#
#     @iron for mi in 1:div(M, m), nj in 1:div(N, n)
#         @cores nj                                     # nj -> the core array
#         @init gemm_zero!(C)
#         @reduce for kk in 1:div(K, k); gemm_acc!(...); end
#     end
#
# Each core gets its own shim tile hosting its operand FIFOs, so the design uses one shim
# per core. That bounds this scheme to the device's shim columns (about 8 on npu2):
# feeding more cores than there are shim tiles would force several cores onto one shim,
# which does not have a DMA channel per core to spare. Scaling to the full array needs the
# L2/MemTile relay (a later increment), which fans out to the cores on-chip.

# Object FIFO depth: how many tiles the shim may buffer ahead of the core. Also the
# ceiling on in-flight host DMA buffer descriptors per operand (see
# `_emit_schedule_runtime!`), which is why it must stay well under the 16-BD-per-tile
# hardware limit.
const FIFO_DEPTH = 2

# The launch behind the `for` form. `space`/`reduction` are tuples of `(name, extent)`;
# `operands` is a tuple of `(direction, NPUArray, access)` where `access` is the tuple of
# axis names indexing that buffer, one per dimension.
function _schedule_launch(
        space, reduction, @nospecialize(init), @nospecialize(step), operands;
        device::AIEDevice = npu2, name::AbstractString = "main",
        flags::AbstractVector{<:AbstractString} = String[], verbose::Bool = false,
        stack_size::Integer = 1024, cores = (),
    )
    extent = Dict{Symbol, Int}()
    for (nm, ex) in space
        extent[nm] = Int(ex)
    end
    for (nm, ex) in reduction
        extent[nm] = Int(ex)
    end
    space_names = Set(nm for (nm, _) in space)
    reduce_names = Set(nm for (nm, _) in reduction)

    # `@cores` splits the space (output-tile) axes into the *spatial* axes -- mapped one
    # tile position per compute core, so those iterations run concurrently across the
    # array -- and the *temporal* axes each core still iterates itself. With no `@cores`
    # every space axis is temporal and the design is single-core, as before.
    core_names = Set(Symbol(a) for a in cores)
    for a in core_names
        a in space_names || error("IRON: `@cores` axis `$a` is not a space axis")
    end
    spatial = Tuple{Symbol, Int}[(nm, extent[nm]) for (nm, _) in space if nm in core_names]
    temporal = Tuple{Symbol, Int}[(nm, extent[nm]) for (nm, _) in space if !(nm in core_names)]
    num_cores = isempty(spatial) ? 1 : prod(e for (_, e) in spatial)

    # Per operand: infer the tile shape from the buffer and its access axes, and pin
    # down its FIFO name, host/kernel-side types, and (for `L2(...)`) the MemTile fan-out
    # pattern -- explicit if given, else inferred from the access: an output joins, an
    # input indexed by a `@cores` axis distributes, one that is not broadcasts.
    specs = map(enumerate(operands)) do (i, op)
        dir, arr, access, l2, pattern, blocks = op
        bufdims = size(arr)
        length(access) == length(bufdims) || error(
            "IRON: operand $i has a $(length(bufdims))-D buffer but $(length(access)) access axes"
        )
        tdims = ntuple(length(bufdims)) do d
            ax = access[d]
            haskey(extent, ax) || error("IRON: operand $i references undeclared axis `$ax`")
            bufdims[d] % extent[ax] == 0 || error(
                "IRON: operand $i dimension $d ($(bufdims[d])) is not divisible by axis `$ax` extent $(extent[ax])"
            )
            bufdims[d] ÷ extent[ax]
        end
        acc = collect(Symbol, access)
        l2pat = if !l2
            nothing
        else
            inferred = dir === :out ? :join :
                any(a -> a in core_names, acc) ? :distribute : :broadcast
            pat = pattern === nothing ? inferred : pattern
            # A join is the output side; broadcast/distribute are the input side.
            (dir === :out) == (pat === :join) || error(
                "IRON: L2 pattern `$pat` does not match a $(dir === :out ? "`Out`" : "`In`") \
                operand (op$i); use `join` for outputs, `broadcast`/`distribute` for inputs"
            )
            pat
        end
        T = eltype(arr)
        # `blocks = (r, s)` streams the DDR tile block-columnar so the kernel loads `Mat`
        # sub-blocks contiguously: the core sees a `(r*s, num_blocks)` tile (each block a
        # column), fed by a `dims_to_stream` off the memtile. Without it the core tile is the
        # DDR tile and no transform is applied.
        blk = blocks === nothing ? nothing : (Int(blocks[1]), Int(blocks[2]))
        core_type, dims = if blk === nothing
            Tile{T, Tuple{tdims...}}, Tuple{Int, Int}[]
        else
            l2 || error("IRON: `blocks` needs `L2(...)` (the transform lives on a memtile)")
            length(tdims) == 2 || error("IRON: `blocks` is only for 2-D operand tiles (op$i)")
            r, s = blk
            nb = (tdims[1] ÷ r) * (tdims[2] ÷ s)
            # Inputs stream block-columnar with the block pattern on their l2l1. Outputs carry
            # the (group-dependent) un-block pattern on l2l3 instead, built in the codegen.
            dims_i = dir === :out ? Tuple{Int, Int}[] : _dims_to_stream_blocks(tdims[1], tdims[2], r, s)
            Tile{T, Tuple{r * s, nb}}, dims_i
        end
        (
            dir = dir, array = arr, access = acc, name = "op$i", l2pattern = l2pat,
            buffer_type = Tile{T, Tuple{bufdims...}}, tile_type = Tile{T, Tuple{tdims...}},
            core_type = core_type, dims = dims, blocks = blk,
        )
    end

    for s in specs
        all(a -> a in space_names || a in reduce_names, s.access) ||
            error("IRON: operand $(s.name) uses an axis that is neither a space nor a reduce axis")
        s.dir === :out && any(a -> a in reduce_names, s.access) && error(
            "IRON: output operand $(s.name) is indexed by a reduce axis; an accumulator \
            must be constant across the reduction (index it by space axes only)"
        )
        # Each core owns one spatial tile position, so cores partition an output only if
        # every spatial axis indexes it -- otherwise two cores would write the same tile.
        s.dir === :out && for a in core_names
            a in s.access || error(
                "IRON: output operand $(s.name) is not indexed by `@cores` axis `$a`; the \
                cores would write the same output tile (index every output by the core axes)"
            )
        end
    end
    any(s -> s.dir === :out, specs) ||
        error("IRON: an @iron reduction needs at least one `Out(...)` operand (the accumulator)")
    any(s -> s.dir === :in, specs) ||
        error("IRON: an @iron reduction needs at least one `In(...)` operand")

    # L2/MemTile forwarding presupposes a core array to fan out to.
    if any(s -> s.l2pattern !== nothing, specs) && num_cores == 1
        error("IRON: `L2(...)` MemTile forwarding needs `@cores` (more than one core)")
    end
    # A broadcast operand must be shared across the cores: it cannot be indexed by a
    # `@cores` axis (each core would then need a different tile, i.e. a distribute).
    for s in specs
        s.l2pattern === :broadcast && any(a -> a in core_names, s.access) && error(
            "IRON: L2 `broadcast` operand $(s.name) is indexed by a `@cores` axis; it is not \
            shared across the cores -- use `distribute` (or drop the explicit pattern)"
        )
        # distribute/join partition the cores along a single `@cores` axis into groups, one
        # memtile each; a second spatial axis (a 2D grid) is not handled yet.
        s.l2pattern in (:distribute, :join) && length(spatial) > 1 && error(
            "IRON: L2 `$(s.l2pattern)` operand $(s.name) needs a single `@cores` axis; \
            distribute/join over a 2D core grid is not implemented yet"
        )
    end
    # Each core group shares one memtile; broadcast operands take their own while columns are
    # free and otherwise fold onto a group memtile. So the design fits in npu2's 8 memtile
    # columns as long as the groups do (or, with no groups, the broadcasts do).
    if any(s -> s.l2pattern !== nothing, specs)
        ngroups = length(_core_groups(num_cores))
        has_dj = any(s -> s.l2pattern in (:distribute, :join), specs)
        nbcast = count(s -> s.l2pattern === :broadcast, specs)
        nmem = has_dj ? ngroups + min(nbcast, max(0, 8 - ngroups)) : nbcast
        nmem <= 8 || error(
            "IRON: this L2 design needs more than npu2's 8 memtile columns \
            ($(has_dj ? "$ngroups core groups" : "$nbcast broadcast operands")); use fewer cores"
        )
    end

    key = (
        typeof(init), typeof(step),
        Tuple(s.buffer_type for s in specs), Tuple(s.tile_type for s in specs),
        Tuple(s.dir for s in specs), Tuple(Tuple(s.access) for s in specs),
        Tuple(s.l2pattern for s in specs), Tuple(s.core_type for s in specs),
        Tuple(Tuple(s.dims) for s in specs), Tuple(s.blocks for s in specs),
        Tuple(spatial), Tuple(temporal), Tuple(reduction), device, String(name), Tuple(flags), Int(stack_size),
    )
    compiled = get!(_LAUNCH_CACHE, key) do
        mlir = _build_schedule_program(init, step, specs, spatial, temporal, reduction, device, name, Int(stack_size))
        compile(mlir, length(specs); flags, verbose)
    end

    run!(compiled, (s.array for s in specs)...)
    return compiled
end

# The tile positions each axis loop takes, as `axis => 0-based value` assignments, in
# column-major order (first axis fastest). No axes yields one empty assignment.
function _axis_coords(axes)
    isempty(axes) && return [Dict{Symbol, Int}()]
    extents = Tuple(Int(e) for (_, e) in axes)
    return [Dict(axes[d][1] => c[d] - 1 for d in eachindex(axes)) for c in CartesianIndices(extents)]
end

# The core loop nest: acquire and initialise the outputs in the outer loop, then acquire
# the inputs and run the step kernel in the inner (reduce) loop, accumulating into the
# outputs held across it. `num_outer` is how many output tiles this core owns -- the
# temporal (non-`@cores`) space positions; the spatial positions are spread over the
# other cores. The core never computes an access pattern -- it consumes tiles in FIFO
# order; the host DMA below feeds the right tile.
function _emit_schedule_core!(ctx::IR.Context, @nospecialize(init), @nospecialize(step), specs, num_outer::Int, num_reduce::Int)
    body = IR.Block(IR.Type[], IR.Location[])
    index = IR.IndexType(; context = ctx)
    const_(v) = (op = arith.constant(; value = IR.Attribute(v, index), location = loc(ctx)); push!(body, op); IR.result(op, 1))
    c0, c1 = const_(0), const_(1)
    cspace, creduce = const_(num_outer), const_(num_reduce)

    inputs = [s for s in specs if s.dir === :in]
    outputs = [s for s in specs if s.dir === :out]
    # The core sees each operand's `core_type` -- the block-columnar L1 tile for a `blocks`
    # operand, the plain tile otherwise (they are equal without `blocks`).
    output_types = Tuple{(s.core_type for s in outputs)...}
    all_types = Tuple{(s.core_type for s in specs)...}

    # Outer (space) loop: one output tile per iteration.
    outer = IR.Block([index], [loc(ctx)])
    out_vals = IR.Value[]
    for s in outputs
        acq = objectfifo_acquire_op(ctx, s.name, Produce, 1, objectfifo_subview_type(ctx, s.core_type))
        push!(outer, acq)
        acc = objectfifo_subview_access_op(ctx, IR.result(acq, 1), 0, memref_type(ctx, s.core_type))
        push!(outer, acc)
        push!(out_vals, IR.result(acc, 1))
    end
    # `@init` is optional: with none, the output tile enters the reduction as-is.
    init === nothing || compile_kernel!(ctx, outer, init, output_types, out_vals)

    # Inner (reduce) loop: acquire the inputs and accumulate into the held outputs.
    inner = IR.Block([index], [loc(ctx)])
    in_vals = IR.Value[]
    for s in inputs
        acq = objectfifo_acquire_op(ctx, s.name, Consume, 1, objectfifo_subview_type(ctx, s.core_type))
        push!(inner, acq)
        acc = objectfifo_subview_access_op(ctx, IR.result(acq, 1), 0, memref_type(ctx, s.core_type))
        push!(inner, acc)
        push!(in_vals, IR.result(acc, 1))
    end
    # The step kernel takes every operand in declaration order; thread the acquired
    # input and held output values back into that order.
    ii, oi = 0, 0
    step_vals = IR.Value[]
    for s in specs
        if s.dir === :in
            ii += 1
            push!(step_vals, in_vals[ii])
        else
            oi += 1
            push!(step_vals, out_vals[oi])
        end
    end
    compile_kernel!(ctx, inner, step, all_types, step_vals)
    for s in inputs
        push!(inner, objectfifo_release_op(ctx, s.name, Consume, 1))
    end
    push!(inner, scf.yield(IR.Value[]; location = loc(ctx)))

    push!(outer, scf.for_(c0, creduce, c1, IR.Value[]; region = region(inner), results = IR.Type[], location = loc(ctx)))
    for s in outputs
        push!(outer, objectfifo_release_op(ctx, s.name, Produce, 1))
    end
    push!(outer, scf.yield(IR.Value[]; location = loc(ctx)))

    push!(body, scf.for_(c0, cspace, c1, IR.Value[]; region = region(outer), results = IR.Type[], location = loc(ctx)))
    push!(body, end_op(ctx))
    return region(body)
end

# The host DMA. One temporal (output-tile) step at a time; within a step feed *every*
# core its reduction inputs and start draining its output, so the cores compute their
# tiles concurrently, then wait on all the outputs. With a single core (`num_cores == 1`,
# no spatial axes) the core loop and coordinate collapse to exactly the earlier
# single-core schedule. Each operand is addressed through its access axes; the spatial
# axes come from the core's coordinate, the temporal and reduce axes from the loops.
function _emit_schedule_runtime!(ctx::IR.Context, specs, spatial, temporal, reduction, num_cores::Int)
    arg_types = IR.Type[memref_type(ctx, s.buffer_type) for s in specs]
    body = IR.Block(arg_types, [loc(ctx) for _ in specs])
    args = IR.Value[IR.argument(body, i) for i in eachindex(specs)]

    function task(i, fname, tiledims, grid; token)
        s = specs[i]
        offset, dims, len = _tile_pattern(size(s.buffer_type), tiledims, grid)
        bd = IR.Block(IR.Type[], IR.Location[])
        push!(bd, dma_bd_op(ctx, args[i], dims, len; offset))
        push!(bd, end_op(ctx))
        t = dma_configure_task_for_op(ctx, fname, region(bd); issue_token = token)
        push!(body, t)
        push!(body, dma_start_task_op(ctx, IR.result(t, 1)))
        return IR.result(t, 1)
    end
    retire(t) = (push!(body, dma_await_task_op(ctx, t)); push!(body, dma_free_task_op(ctx, t)))

    core_names = Set(nm for (nm, _) in spatial)
    grid_of(s, coord) = Tuple(coord[a] for a in s.access)
    # A group's super-tile is the `gidx`-th tile of `gsize` core slices along the `@cores`
    # axes, so its grid index there is the group index and the loop coordinate elsewhere.
    group_grid(s, coord, gidx) = Tuple(a in core_names ? gidx : coord[a] for a in s.access)

    # Shared inputs (broadcast/distribute) cross DDR once into the shim->memtile FIFO; the
    # memtile fans them out. Distribute moves a group's super-tile, broadcast a plain tile.
    # Direct inputs are fed per core; join outputs are drained per group as super-tiles.
    bcast_in = [i for (i, s) in enumerate(specs) if s.dir === :in && s.l2pattern === :broadcast]
    dist_in = [i for (i, s) in enumerate(specs) if s.dir === :in && s.l2pattern === :distribute]
    direct_in = [i for (i, s) in enumerate(specs) if s.dir === :in && s.l2pattern === nothing]
    join_out = [i for (i, s) in enumerate(specs) if s.dir === :out && s.l2pattern === :join]
    direct_out = [i for (i, s) in enumerate(specs) if s.dir === :out && s.l2pattern === nothing]
    core_coords = _axis_coords(spatial)   # one spatial coordinate per core
    groups = _core_groups(num_cores)

    for tc in _axis_coords(temporal)
        # A sliding window of in-flight input BDs, one queue per FIFO -- per (operand, core)
        # for direct inputs, per (operand, group) for the shared broadcast/distribute ones.
        # The shim runs at most `FIFO_DEPTH` tiles ahead of a core before its object FIFO
        # backpressures, so configuring more up front buys no overlap and just burns buffer
        # descriptors (a tile holds at most 16). Await + free the oldest before the next.
        inflight = Dict{Any, Vector{IR.Value}}()
        for i in bcast_in
            inflight[(:shared, i, 0)] = IR.Value[]
        end
        for i in dist_in, gidx in eachindex(groups)
            inflight[(:shared, i, gidx)] = IR.Value[]
        end
        for i in direct_in, c in 0:(num_cores - 1)
            inflight[(i, c)] = IR.Value[]
        end

        # Advance the reduction in lockstep across the cores: feed *every* core its rc-th
        # input tiles before moving to rc+1. The window's await then paces all cores
        # together, so they compute concurrently -- feeding one core's whole reduction
        # first would gate the next core on this core's progress and serialise them.
        for rc in _axis_coords(reduction)
            for i in bcast_in                       # shared tile, once for all cores
                q = inflight[(:shared, i, 0)]
                length(q) >= FIFO_DEPTH && retire(popfirst!(q))
                push!(q, task(i, "op$(i)_l3l2", size(specs[i].tile_type), grid_of(specs[i], merge(tc, rc)); token = true))
            end
            for i in dist_in, (gidx, group) in enumerate(groups)   # a group's super-tile, sliced per core by the link
                q = inflight[(:shared, i, gidx)]
                length(q) >= FIFO_DEPTH && retire(popfirst!(q))
                push!(q, task(i, "op$(i)_l3l2_g$(gidx - 1)", _super_dims(specs[i], core_names, length(group)), group_grid(specs[i], merge(tc, rc), gidx - 1); token = true))
            end
            for (c, sc) in enumerate(core_coords)
                cidx = c - 1
                full = merge(merge(sc, tc), rc)
                for i in direct_in
                    q = inflight[(i, cidx)]
                    length(q) >= FIFO_DEPTH && retire(popfirst!(q))
                    push!(q, task(i, _fifo_name(i, cidx, num_cores), size(specs[i].tile_type), grid_of(specs[i], full); token = true))
                end
            end
        end

        # Drain each core's output -- per core for direct outputs, per group as a super-tile
        # for a join -- wait on them all, then retire the trailing inputs.
        outs = IR.Value[]
        for i in join_out, (gidx, group) in enumerate(groups)
            push!(outs, task(i, "op$(i)_l2l3_g$(gidx - 1)", _super_dims(specs[i], core_names, length(group)), group_grid(specs[i], tc, gidx - 1); token = true))
        end
        for (c, sc) in enumerate(core_coords)
            cidx = c - 1
            for i in direct_out
                push!(outs, task(i, _fifo_name(i, cidx, num_cores), size(specs[i].tile_type), grid_of(specs[i], merge(sc, tc)); token = true))
            end
        end
        for o in outs
            push!(body, dma_await_task_op(ctx, o))
        end
        for i in bcast_in, t in inflight[(:shared, i, 0)]
            retire(t)
        end
        for i in dist_in, gidx in eachindex(groups), t in inflight[(:shared, i, gidx)]
            retire(t)
        end
        for c in 0:(num_cores - 1), i in direct_in, t in inflight[(i, c)]
            retire(t)
        end
    end
    return runtime_sequence_op(ctx, "sequence", region(body))
end

# The object FIFO for operand `i` feeding/draining core `c` (0-based). With a single
# core the plain `op$i` name keeps the design byte-identical to the pre-`@cores` one.
_fifo_name(i, c, num_cores) = num_cores == 1 ? "op$i" : "op$(i)_c$c"

_retile(::Type{Tile{T, D}}, dims) where {T, D} = Tile{T, Tuple{dims...}}

# The `dims_to_stream` pattern that rearranges a column-major `(m, k)` memtile tile into a
# block-columnar L1 layout: each `r`x`s` matmul block laid out contiguously in column-major
# order (matching the whole-tile `Mat` convention verified in matmul_tile.jl), blocks in
# `(mb outer, kb inner)` order. The memtile holds element `(i, j)` at offset `j*m + i`, so the
# nested read `mb, kb, c, rr` gives these `(size, stride)` pairs (outermost first). The L1
# tile is then `Tile{T, (r*s, (m/r)*(k/s))}`, block `b` its `b`-th column.
function _dims_to_stream_blocks(m::Int, k::Int, r::Int, s::Int)
    (m % r == 0 && k % s == 0) || error(
        "IRON: matmul block $(r)x$(s) does not tile a $(m)x$(k) operand tile"
    )
    return Tuple{Int, Int}[(m ÷ r, r), (k ÷ s, s * m), (s, m), (r, 1)]
end

# The inverse, for a joined output: read one core's block-columnar `(m, n)` tile (each `rxt`
# block a column, blocks in mb-outer/nb-inner order) out of the memtile and emit it
# column-major, back to the DDR order. Applied per join input; the link's `src_offsets`
# concatenate the cores. Exactly four DMA dimensions, the memtile's limit.
function _dims_to_stream_unblock(m::Int, n::Int, r::Int, t::Int)
    (m % r == 0 && n % t == 0) || error(
        "IRON: matmul block $(r)x$(t) does not tile output tile $(m)x$(n)"
    )
    return Tuple{Int, Int}[(n ÷ t, r * t), (t, r), (m ÷ r, (n ÷ t) * r * t), (r, 1)]
end

# A single memtile can only fan out to (in from) one FIFO per core over its DMA channels,
# so distribute/join partition the cores into groups of at most `L2_GROUP`, one memtile per
# group. This matches the npu2 column shape (4 compute rows per memtile).
const L2_GROUP = 4

# The 0-based core indices in each group, in order.
function _core_groups(num_cores)
    num_cores <= L2_GROUP && return [collect(0:(num_cores - 1))]
    num_cores % L2_GROUP == 0 || error(
        "IRON: with L2 distribute/join the core count ($num_cores) must be a multiple of \
        L2_GROUP ($L2_GROUP)"
    )
    return [collect(g:(g + L2_GROUP - 1)) for g in 0:L2_GROUP:(num_cores - 1)]
end

# The MemTile-level "super-tile" one group's memtile moves in a single DDR<->L2 DMA: it
# spans `gsize` core slices along the `@cores` axes and one temporal/reduce position
# elsewhere. The link then slices it per core.
function _super_dims(s, core_names, gsize)
    tdims = size(s.tile_type)
    ntuple(length(tdims)) do d
        s.access[d] in core_names ? gsize * tdims[d] : tdims[d]
    end
end

# Column-major element offset of a core's sub-tile within its group's super-tile, given the
# group's spatial `base` coordinate -- the `src_offsets`/`dst_offsets` the link needs.
function _sub_offset(s, sc, core_names, gsize, base)
    superd, tdims = _super_dims(s, core_names, gsize), size(s.tile_type)
    off, stride = 0, 1
    for d in eachindex(superd)
        a = s.access[d]
        pos = a in core_names ? (sc[a] - get(base, a, 0)) * tdims[d] : 0
        off += pos * stride
        stride *= superd[d]
    end
    return off
end

function _build_schedule_program(
        @nospecialize(init), @nospecialize(step), specs, spatial, temporal, reduction,
        device::AIEDevice, name::AbstractString, stack_size::Int; ctx::IR.Context = context(),
    )
    num_temporal = isempty(temporal) ? 1 : prod(Int(e) for (_, e) in temporal)
    num_reduce = isempty(reduction) ? 1 : prod(Int(e) for (_, e) in reduction)
    num_cores = isempty(spatial) ? 1 : prod(Int(e) for (_, e) in spatial)

    device_body = IR.Block(IR.Type[], IR.Location[])

    if num_cores == 1
        # Single core: one shim tile per operand, exactly as the pre-`@cores` design (the
        # tile order and `op$i` FIFO names are preserved so the emitted module is
        # unchanged for existing single-core schedules).
        core = logical_tile_op(ctx, CoreTile)
        push!(device_body, core)
        core_tile = IR.result(core, 1)
        shims = IR.Value[]
        for _ in specs
            t = logical_tile_op(ctx, ShimNOCTile)
            push!(device_body, t)
            push!(shims, IR.result(t, 1))
        end
        for (i, s) in enumerate(specs)
            into_core, from_core = shims[i], core_tile
            producer_tile, consumer_tile = s.dir === :in ? (into_core, from_core) : (from_core, into_core)
            push!(device_body, objectfifo_op(ctx, s.name, producer_tile, IR.Value[consumer_tile], objectfifo_type(ctx, s.tile_type), FIFO_DEPTH))
        end
        push!(device_body, core_op(ctx, core_tile, _emit_schedule_core!(ctx, init, step, specs, num_temporal, num_reduce); stack_size))
    else
        # Multi-core. Create every compute tile first so L2 FIFOs can name them as
        # producers/consumers.
        core_names = Set(nm for (nm, _) in spatial)
        core_coords = _axis_coords(spatial)
        core_tiles = IR.Value[]
        for _ in 0:(num_cores - 1)
            t = logical_tile_op(ctx, CoreTile)
            push!(device_body, t)
            push!(core_tiles, IR.result(t, 1))
        end

        # The FIFO each core reads/writes for each operand; L2 operands fill it below,
        # direct operands get a per-core shim FIFO afterwards.
        corefifo = [Dict{Int, String}() for _ in 1:num_cores]

        # L2 operands route DDR<->shim<->MemTile<->cores through linked FIFOs sharing a
        # memtile (`aie.objectfifo.link`), so the operand crosses DDR once and fans out/in
        # on-chip instead of using a shim channel per core:
        #   broadcast  -- shared input: one memtile, l3l2 -> l2l1 with every core a consumer;
        #   distribute -- per-core input: l3l2 (a group's super-tile) -> one l2l1 per core,
        #                 dst_offsets slicing each core's tile out;
        #   join       -- per-core output: one l1l2 per core -> l2l3 (super-tile), src_offsets.
        # All the distribute/join FIFOs for a group of cores share that group's ONE memtile
        # (as whole_array's per-column memtile does), so the memtile count is the group count,
        # not the operand count -- which is what lets 32 cores fit in 8 memtile columns.
        groups = _core_groups(num_cores)
        djspecs = [(i, s) for (i, s) in enumerate(specs) if s.l2pattern in (:distribute, :join)]
        group_mems = IR.Value[]
        for (gidx, group) in enumerate(groups)
            isempty(djspecs) && break
            gsize = length(group)
            base = Dict(a => (gidx - 1) * L2_GROUP for a in core_names)
            mem = logical_tile_op(ctx, MemTile); push!(device_body, mem)
            shim = logical_tile_op(ctx, ShimNOCTile); push!(device_body, shim)
            mem_tile, shim_tile = IR.result(mem, 1), IR.result(shim, 1)
            push!(group_mems, mem_tile)
            for (i, s) in djspecs
                core_of = objectfifo_type(ctx, s.core_type)   # block-columnar for `blocks`
                super_of = objectfifo_type(ctx, _retile(s.tile_type, _super_dims(s, core_names, gsize)))
                offs = [_sub_offset(s, core_coords[c + 1], core_names, gsize, base) for c in group]
                if s.l2pattern === :distribute
                    l3l2 = "op$(i)_l3l2_g$(gidx - 1)"
                    push!(device_body, objectfifo_op(ctx, l3l2, shim_tile, IR.Value[mem_tile], super_of, FIFO_DEPTH))
                    outs = String[]
                    for c in group
                        l2l1 = "op$(i)_l2l1_c$(c)"
                        push!(device_body, objectfifo_op(ctx, l2l1, mem_tile, IR.Value[core_tiles[c + 1]], core_of, FIFO_DEPTH; dims_to_stream = s.dims))
                        push!(outs, l2l1)
                        corefifo[c + 1][i] = l2l1
                    end
                    push!(device_body, objectfifo_link_op(ctx, [l3l2], outs; dst_offsets = offs))
                else # :join
                    l2l3 = "op$(i)_l2l3_g$(gidx - 1)"
                    # With `blocks`, the cores write block-columnar tiles (core_type on l1l2);
                    # l2l3 carries the un-block pattern that restores the DDR (m, n*gsize) order.
                    l2l3_dims = if s.blocks === nothing
                        Tuple{Int, Int}[]
                    else
                        md, nd = size(s.tile_type)
                        _dims_to_stream_unblock(md, nd, s.blocks[1], s.blocks[2])
                    end
                    push!(device_body, objectfifo_op(ctx, l2l3, mem_tile, IR.Value[shim_tile], super_of, FIFO_DEPTH; dims_to_stream = l2l3_dims))
                    ins = String[]
                    for c in group
                        l1l2 = "op$(i)_l1l2_c$(c)"
                        push!(device_body, objectfifo_op(ctx, l1l2, core_tiles[c + 1], IR.Value[mem_tile], core_of, FIFO_DEPTH))
                        push!(ins, l1l2)
                        corefifo[c + 1][i] = l1l2
                    end
                    push!(device_body, objectfifo_link_op(ctx, ins, [l2l3]; src_offsets = offs))
                end
            end
        end

        # Broadcast operands multicast to every core (only 1 in + 1 out on the memtile, so
        # cheap). Give each its own memtile while columns are free; once the groups use all 8
        # columns (the 32-core case), fold the broadcast onto a group memtile round-robin.
        bcast_i = 0
        for (i, s) in enumerate(specs)
            s.l2pattern === :broadcast || continue
            tile_of = objectfifo_type(ctx, s.tile_type)
            core_of = objectfifo_type(ctx, s.core_type)
            shim = logical_tile_op(ctx, ShimNOCTile); push!(device_body, shim)
            mem_tile = if length(group_mems) + bcast_i < 8
                mem = logical_tile_op(ctx, MemTile); push!(device_body, mem)
                IR.result(mem, 1)
            else
                group_mems[(bcast_i % length(group_mems)) + 1]
            end
            bcast_i += 1
            l3l2, l2l1 = "op$(i)_l3l2", "op$(i)_l2l1"
            push!(device_body, objectfifo_op(ctx, l3l2, IR.result(shim, 1), IR.Value[mem_tile], tile_of, FIFO_DEPTH))
            # The l3l2 delivers the DDR tile to the memtile; `dims_to_stream` on the
            # memtile->core l2l1 rearranges it into the block-columnar core layout (empty and
            # so a plain relay when the operand has no `blocks`).
            push!(device_body, objectfifo_op(ctx, l2l1, mem_tile, core_tiles, core_of, FIFO_DEPTH; dims_to_stream = s.dims))
            push!(device_body, objectfifo_link_op(ctx, [l3l2], [l2l1]))
            for c in 1:num_cores
                corefifo[c][i] = l2l1
            end
        end

        # Per core: a shim tile hosting its *direct* (non-L2) operand FIFOs -- only if it has
        # any. So the shim count is at most one per core (plus one per L2 operand), bounded
        # by the device's shim columns; with every operand on L2 the cores need no shim.
        for c in 0:(num_cores - 1)
            core_tile = core_tiles[c + 1]
            direct = [i for (i, s) in enumerate(specs) if s.l2pattern === nothing]
            shim_tile = if isempty(direct)
                nothing
            else
                shim = logical_tile_op(ctx, ShimNOCTile)
                push!(device_body, shim)
                IR.result(shim, 1)
            end
            core_specs = map(enumerate(specs)) do (i, s)
                haskey(corefifo[c + 1], i) && return merge(s, (; name = corefifo[c + 1][i]))
                fname = _fifo_name(i, c, num_cores)
                producer_tile, consumer_tile = s.dir === :in ? (shim_tile, core_tile) : (core_tile, shim_tile)
                push!(device_body, objectfifo_op(ctx, fname, producer_tile, IR.Value[consumer_tile], objectfifo_type(ctx, s.tile_type), FIFO_DEPTH))
                merge(s, (; name = fname))
            end
            push!(device_body, core_op(ctx, core_tile, _emit_schedule_core!(ctx, init, step, core_specs, num_temporal, num_reduce); stack_size))
        end
    end

    push!(device_body, _emit_schedule_runtime!(ctx, specs, spatial, temporal, reduction, num_cores))
    push!(device_body, end_op(ctx))

    mod = IR.Module(loc(ctx))
    push!(IR.body(mod), device_op(ctx, device, name, region(device_body)))
    IR.verify(IR.Operation(mod)) ||
        error("IRON: the @iron reduction generated an invalid MLIR module (see the diagnostics above)")
    canonicalize!(mod, ctx)
    return string(IR.Operation(mod))
end

# --- `@iron for` front end ---------------------------------------------------
# Turns the `for` form of `@iron` (parsed in `launch.jl`) into a `_schedule_launch`
# call. The outer `for` header gives the space (output-tile) axes; the body holds an
# optional `@init` and the reduction, written as a nested `@reduce for` loop whose own
# header is the reduction axes and whose body is the step call:
#
#     @iron for mi in 1:div(M, m), nj in 1:div(N, n)
#         @init gemm_zero!(C)
#         @reduce for kk in 1:div(K, k)
#             gemm_acc!(In(A)[mi, kk], In(B)[kk, nj], Out(C)[mi, nj])
#         end
#     end
#
# A design with no reduction writes the step call directly in the space body instead.

# `[(:name, extent_expr), ...]` for the space or reduction axes.
_axes_to_expr(axes) = Expr(:tuple, [Expr(:tuple, QuoteNode(nm), esc(ex)) for (nm, ex) in axes]...)

# A `for` header -- `for mi in 1:M÷m, nj in 1:N÷n` -> [(:mi, :(length(1:M÷m))), ...].
# One spec is a bare `=`; several are a `:block` of them. Each axis extent is the
# length of its range, evaluated at launch. `what` names the axis kind for errors.
function _parse_for_axes(header, what)
    specs = Meta.isexpr(header, :block) ? header.args : Any[header]
    axes = Tuple{Symbol, Any}[]
    for s in specs
        s isa LineNumberNode && continue
        (Meta.isexpr(s, :(=)) && s.args[1] isa Symbol) ||
            error("@iron: a $what axis must be written `var in range`, got `$s`")
        push!(axes, (s.args[1], Expr(:call, :length, s.args[2])))
    end
    isempty(axes) && error("@iron: the `for` needs at least one $what axis")
    return axes
end

# One step-call operand: `In(a)[mi, kk]` / `Out(c)[mi, nj]` -> (dir, arr, [axes...], l2,
# pattern). An operand may be wrapped in `L2(...)` to route it through a MemTile; the
# fan-out pattern (broadcast/distribute/join) is inferred from the access axes unless
# stated explicitly, as `L2(In(a); broadcast)` or `L2(In(a), broadcast)`.
const _L2_PATTERNS = (:broadcast, :distribute, :join)

function _parse_operand(op)
    Meta.isexpr(op, :ref) || error(
        "@iron: each step argument must be `In(a)[axes...]`, `Out(a)[axes...]` or `L2(...)[axes...]`, got `$op`"
    )
    callee, access = op.args[1], op.args[2:end]

    # Peel an optional `L2(...)` wrapper, collecting an explicit pattern and/or a
    # `blocks = (r, s)` matmul-tile shape (which streams the tile block-columnar via
    # `dims_to_stream`, so a kernel can load `Mat{r,s}` sub-blocks contiguously).
    l2, pattern, blocks = false, nothing, nothing
    parse_param(p) =
        if p isa Symbol && p in _L2_PATTERNS
            pattern = p
        elseif Meta.isexpr(p, :kw) && p.args[1] === :blocks
            blocks = p.args[2]
        else
            error("@iron: `L2(...)` takes a pattern (broadcast/distribute/join) and/or \
            `blocks = (r, s)`, got `$p`")
        end
    if Meta.isexpr(callee, :call) && callee.args[1] === :L2
        l2 = true
        for a in callee.args[2:end]
            if Meta.isexpr(a, :parameters)                 # `L2(In(a); broadcast, blocks=(4,8))`
                foreach(parse_param, a.args)
            elseif a in _L2_PATTERNS                        # `L2(In(a), broadcast)`
                pattern = a
            elseif Meta.isexpr(a, :call) && a.args[1] in (:In, :Out)
                callee = a
            else
                error("@iron: `L2(...)` wraps one `In(a)`/`Out(a)` (optionally with a pattern), got `$op`")
            end
        end
    end

    (Meta.isexpr(callee, :call) && length(callee.args) == 2 && callee.args[1] in (:In, :Out)) ||
        error("@iron: a step argument must wrap the buffer in `In(...)` or `Out(...)`, got `$op`")
    all(a -> a isa Symbol, access) ||
        error("@iron: the `[...]` of a step argument must be axis names, got `$op`")
    dir = callee.args[1] === :In ? :in : :out
    return (dir, callee.args[2], Symbol[access...], l2, pattern, blocks)
end

# The single step call inside a `@reduce for ... end` (or the space body, no reduction)
# -> (step_kernel, operands). Only the call is allowed; the loop nest itself is what the
# schedule generates.
function _parse_step_call(stmt)
    Meta.isexpr(stmt, :call) ||
        error("@iron: expected a step call `kernel(In(a)[...], Out(c)[...])`, got `$stmt`")
    return stmt.args[1], map(_parse_operand, stmt.args[2:end])
end

function _parse_step_body(body)
    Meta.isexpr(body, :block) || return _parse_step_call(body)
    calls = filter(s -> !(s isa LineNumberNode), body.args)
    length(calls) == 1 ||
        error("@iron: the `@reduce for` body must hold exactly one step call, got $(length(calls)) statements")
    return _parse_step_call(calls[1])
end

# Build the `_schedule_launch` call from the `@iron for ... end` target and the
# already-parsed `key = value` options (see the `@iron` macro in `launch.jl`).
function _build_iron_schedule(forexpr, kws)
    header, block = forexpr.args[1], forexpr.args[2]
    Meta.isexpr(block, :block) || error("@iron: malformed `for` body")
    space = _parse_for_axes(header, "space")

    init = step = operands = nothing
    reduction = Tuple{Symbol, Any}[]
    coreaxes = Symbol[]
    for stmt in block.args
        stmt isa LineNumberNode && continue
        if Meta.isexpr(stmt, :macrocall) && stmt.args[1] === Symbol("@init")
            init === nothing || error("@iron: more than one `@init` in the `for` body")
            initexpr = stmt.args[end]
            init = Meta.isexpr(initexpr, :call) ? initexpr.args[1] : initexpr
        elseif Meta.isexpr(stmt, :macrocall) && stmt.args[1] === Symbol("@cores")
            isempty(coreaxes) || error("@iron: more than one `@cores` directive in the `for` body")
            for p in stmt.args[3:end]   # args[2] is the LineNumberNode
                if p isa Symbol
                    push!(coreaxes, p)
                elseif Meta.isexpr(p, :tuple) && all(a -> a isa Symbol, p.args)
                    append!(coreaxes, p.args)
                else
                    error("@iron: `@cores` takes space axis names, as in `@cores nj`, got `$p`")
                end
            end
            isempty(coreaxes) && error("@iron: `@cores` needs at least one axis name")
        elseif Meta.isexpr(stmt, :macrocall) && stmt.args[1] === Symbol("@reduce")
            operands === nothing || error("@iron: the `for` body declares its step more than once")
            redfor = stmt.args[end]
            Meta.isexpr(redfor, :for) || error(
                "@iron: `@reduce` must wrap a `for` loop, as in `@reduce for kk in 1:div(K, k) … end`"
            )
            reduction = _parse_for_axes(redfor.args[1], "reduction")
            step, operands = _parse_step_body(redfor.args[2])
        elseif Meta.isexpr(stmt, :call)
            operands === nothing || error("@iron: the `for` body declares its step more than once")
            step, operands = _parse_step_call(stmt)
        else
            error("@iron: unexpected line in the `for` body: `$stmt`")
        end
    end

    step === nothing && error(
        "@iron: the `for` body needs a step -- a `@reduce for … end` whose body is a call \
        `kernel(In(a)[…], Out(c)[…])` (or that call directly, for no reduction)"
    )
    isempty(operands) && error("@iron: the step call declares no operands")

    space_syms = Set(nm for (nm, _) in space)
    for a in coreaxes
        a in space_syms ||
            error("@iron: `@cores` axis `$a` is not one of the `for` (space) axes")
    end

    ops_expr = Expr(:tuple, [
        Expr(:tuple, QuoteNode(dir), esc(arr), Expr(:tuple, QuoteNode.(access)...),
             l2, pattern === nothing ? :nothing : QuoteNode(pattern),
             blocks === nothing ? :nothing : esc(blocks))
        for (dir, arr, access, l2, pattern, blocks) in operands
    ]...)

    # `@cores` becomes a `cores = (:axis, ...)` keyword on the launch call, alongside the
    # user's own options in `kws`.
    allkws = copy(kws)
    isempty(coreaxes) ||
        push!(allkws, Expr(:kw, :cores, Expr(:tuple, QuoteNode.(coreaxes)...)))

    return Expr(
        :call, _schedule_launch,
        _axes_to_expr(space), _axes_to_expr(reduction),
        init === nothing ? :nothing : esc(init), esc(step), ops_expr,
        allkws...,
    )
end
