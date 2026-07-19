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
    # down its FIFO name and host/kernel-side types.
    specs = map(enumerate(operands)) do (i, op)
        dir, arr, access = op
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
        T = eltype(arr)
        (
            dir = dir, array = arr, access = collect(Symbol, access), name = "op$i",
            buffer_type = Tile{T, Tuple{bufdims...}}, tile_type = Tile{T, Tuple{tdims...}},
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

    key = (
        typeof(init), typeof(step),
        Tuple(s.buffer_type for s in specs), Tuple(s.tile_type for s in specs),
        Tuple(s.dir for s in specs), Tuple(Tuple(s.access) for s in specs),
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
    output_types = Tuple{(s.tile_type for s in outputs)...}
    all_types = Tuple{(s.tile_type for s in specs)...}

    # Outer (space) loop: one output tile per iteration.
    outer = IR.Block([index], [loc(ctx)])
    out_vals = IR.Value[]
    for s in outputs
        acq = objectfifo_acquire_op(ctx, s.name, Produce, 1, objectfifo_subview_type(ctx, s.tile_type))
        push!(outer, acq)
        acc = objectfifo_subview_access_op(ctx, IR.result(acq, 1), 0, memref_type(ctx, s.tile_type))
        push!(outer, acc)
        push!(out_vals, IR.result(acc, 1))
    end
    # `@init` is optional: with none, the output tile enters the reduction as-is.
    init === nothing || compile_kernel!(ctx, outer, init, output_types, out_vals)

    # Inner (reduce) loop: acquire the inputs and accumulate into the held outputs.
    inner = IR.Block([index], [loc(ctx)])
    in_vals = IR.Value[]
    for s in inputs
        acq = objectfifo_acquire_op(ctx, s.name, Consume, 1, objectfifo_subview_type(ctx, s.tile_type))
        push!(inner, acq)
        acc = objectfifo_subview_access_op(ctx, IR.result(acq, 1), 0, memref_type(ctx, s.tile_type))
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

    function task(i, fname, grid; token)
        s = specs[i]
        offset, dims, len = _tile_pattern(size(s.buffer_type), size(s.tile_type), grid)
        bd = IR.Block(IR.Type[], IR.Location[])
        push!(bd, dma_bd_op(ctx, args[i], dims, len; offset))
        push!(bd, end_op(ctx))
        t = dma_configure_task_for_op(ctx, fname, region(bd); issue_token = token)
        push!(body, t)
        push!(body, dma_start_task_op(ctx, IR.result(t, 1)))
        return IR.result(t, 1)
    end
    retire(t) = (push!(body, dma_await_task_op(ctx, t)); push!(body, dma_free_task_op(ctx, t)))

    grid_of(s, coord) = Tuple(coord[a] for a in s.access)
    in_ops = [i for (i, s) in enumerate(specs) if s.dir === :in]
    out_ops = [i for (i, s) in enumerate(specs) if s.dir === :out]
    core_coords = _axis_coords(spatial)   # one spatial coordinate per core

    for tc in _axis_coords(temporal)
        # A sliding window of in-flight input BDs, one queue per (operand, core) FIFO. The
        # shim runs at most `FIFO_DEPTH` tiles ahead of a core before its object FIFO
        # backpressures, so configuring more up front buys no overlap and just burns buffer
        # descriptors (a tile holds at most 16). Await + free the oldest before the next.
        inflight = Dict((i, c) => IR.Value[] for i in in_ops, c in 0:(num_cores - 1))

        # Advance the reduction in lockstep across the cores: feed *every* core its rc-th
        # input tiles before moving to rc+1. The window's await then paces all cores
        # together, so they compute concurrently -- feeding one core's whole reduction
        # first would gate the next core on this core's progress and serialise them.
        for rc in _axis_coords(reduction)
            for (c, sc) in enumerate(core_coords)
                cidx = c - 1
                full = merge(merge(sc, tc), rc)
                for i in in_ops
                    q = inflight[(i, cidx)]
                    length(q) >= FIFO_DEPTH && retire(popfirst!(q))
                    push!(q, task(i, _fifo_name(i, cidx, num_cores), grid_of(specs[i], full); token = true))
                end
            end
        end

        # Start and wait on every core's output tile, then retire the trailing inputs.
        outs = IR.Value[]
        for (c, sc) in enumerate(core_coords)
            cidx = c - 1
            for i in out_ops
                push!(outs, task(i, _fifo_name(i, cidx, num_cores), grid_of(specs[i], merge(sc, tc)); token = true))
            end
        end
        for o in outs
            push!(body, dma_await_task_op(ctx, o))
        end
        for c in 0:(num_cores - 1), i in in_ops, t in inflight[(i, c)]
            retire(t)
        end
    end
    return runtime_sequence_op(ctx, "sequence", region(body))
end

# The object FIFO for operand `i` feeding/draining core `c` (0-based). With a single
# core the plain `op$i` name keeps the design byte-identical to the pre-`@cores` one.
_fifo_name(i, c, num_cores) = num_cores == 1 ? "op$i" : "op$(i)_c$c"

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
        # Multi-core: one shim tile *per core*, hosting that core's operand FIFOs. So the
        # shim count is the core count (bounded by the device's shim columns), and each
        # shim needs only one DMA channel per operand -- not one channel per core, which a
        # single shared shim tile could not supply. Each core runs the same program over
        # its temporal slice, reading/writing its own FIFOs.
        for c in 0:(num_cores - 1)
            core = logical_tile_op(ctx, CoreTile)
            push!(device_body, core)
            core_tile = IR.result(core, 1)
            shim = logical_tile_op(ctx, ShimNOCTile)
            push!(device_body, shim)
            shim_tile = IR.result(shim, 1)
            core_specs = map(enumerate(specs)) do (i, s)
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

# One step-call operand: `In(a)[mi, kk]` / `Out(c)[mi, nj]` -> (dir, arr, [axes...]).
function _parse_operand(op)
    Meta.isexpr(op, :ref) || error(
        "@iron: each step argument must be `In(a)[axes...]` or `Out(a)[axes...]`, got `$op`"
    )
    callee, access = op.args[1], op.args[2:end]
    (Meta.isexpr(callee, :call) && length(callee.args) == 2 && callee.args[1] in (:In, :Out)) ||
        error("@iron: a step argument must wrap the buffer in `In(...)` or `Out(...)`, got `$op`")
    all(a -> a isa Symbol, access) ||
        error("@iron: the `[...]` of a step argument must be axis names, got `$op`")
    dir = callee.args[1] === :In ? :in : :out
    return (dir, callee.args[2], Symbol[access...])
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
        Expr(:tuple, QuoteNode(dir), esc(arr), Expr(:tuple, QuoteNode.(access)...))
        for (dir, arr, access) in operands
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
