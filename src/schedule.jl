# A schedule DSL for tiled *reductions* -- the shape `@iron` cannot infer. A GEMM is
# the archetype: an output tile is held across an inner loop that streams the input
# operands and accumulates into it, and each operand's tile is addressed by a
# different mix of the loop variables. That schedule is not visible in the kernels,
# so it is declared here:
#
#     @iron_schedule begin
#         space  = (mi = M÷m, nj = N÷n)   # the output-tile iteration
#         reduce = (kk = K÷k,)            # the accumulation axis
#         init   = gemm_zero!             # run on the accumulator once per output tile
#         step   = gemm_acc!              # run each reduction step (C += A*B)
#         In(A)  => (mi, kk)              # each operand: direction and the axes that
#         In(B)  => (kk, nj)              # index its tile, in buffer-dimension order
#         Out(C) => (mi, nj)             # (the Out operand is the accumulator)
#     end
#
# This generalises the hand-written generator in `gemm.jl`: the core loop nest is the
# same `emit_gemm_core!` shape (outputs acquired and initialised in the outer/space
# loop, inputs acquired and reduced in the inner/reduce loop), and the host DMA is the
# same `emit_gemm_runtime!` shape (per output tile, stream the reduction's input tiles,
# then drain the output). A tile shape is not annotated -- it follows from the buffer
# shape and the extents of the axes indexing it: buffer dimension `d` of extent `D`,
# indexed by an axis of extent `E`, gives a tile extent `D ÷ E`.

# The launch behind `@iron_schedule`. `space`/`reduction` are tuples of `(name, extent)`;
# `operands` is a tuple of `(direction, NPUArray, access)` where `access` is the tuple of
# axis names indexing that buffer, one per dimension.
function _schedule_launch(
        space, reduction, @nospecialize(init), @nospecialize(step), operands;
        device::AIEDevice = npu2, name::AbstractString = "main",
        flags::AbstractVector{<:AbstractString} = String[], verbose::Bool = false,
        stack_size::Integer = 1024,
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
    end
    any(s -> s.dir === :out, specs) ||
        error("IRON: @iron_schedule needs at least one `Out(...)` operand (the accumulator)")
    any(s -> s.dir === :in, specs) ||
        error("IRON: @iron_schedule needs at least one `In(...)` operand")

    key = (
        typeof(init), typeof(step),
        Tuple(s.buffer_type for s in specs), Tuple(s.tile_type for s in specs),
        Tuple(s.dir for s in specs), Tuple(Tuple(s.access) for s in specs),
        Tuple(space), Tuple(reduction), device, String(name), Tuple(flags), Int(stack_size),
    )
    compiled = get!(_LAUNCH_CACHE, key) do
        mlir = _build_schedule_program(init, step, specs, space, reduction, device, name, Int(stack_size))
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

# The core loop nest, generalising `emit_gemm_core!`: acquire and initialise the
# outputs in the outer (space) loop, then acquire the inputs and run the step kernel in
# the inner (reduce) loop, accumulating into the outputs held across it. The core never
# computes an access pattern -- it consumes tiles in FIFO order; the host DMA below
# feeds the right tile.
function _emit_schedule_core!(ctx::IR.Context, @nospecialize(init), @nospecialize(step), specs, num_space::Int, num_reduce::Int)
    body = IR.Block(IR.Type[], IR.Location[])
    index = IR.IndexType(; context = ctx)
    const_(v) = (op = arith.constant(; value = IR.Attribute(v, index), location = loc(ctx)); push!(body, op); IR.result(op, 1))
    c0, c1 = const_(0), const_(1)
    cspace, creduce = const_(num_space), const_(num_reduce)

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
    compile_kernel!(ctx, outer, init, output_types, out_vals)

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

# The host DMA, generalising `emit_gemm_runtime!`: for each output tile (a space
# coordinate), stream every input tile the reduction reads -- addressing each operand
# through its access axes -- then drain the output tile, waiting on it before the next.
function _emit_schedule_runtime!(ctx::IR.Context, specs, space, reduction)
    arg_types = IR.Type[memref_type(ctx, s.buffer_type) for s in specs]
    body = IR.Block(arg_types, [loc(ctx) for _ in specs])
    args = IR.Value[IR.argument(body, i) for i in eachindex(specs)]

    function task(i, grid; token)
        s = specs[i]
        offset, dims, len = _tile_pattern(size(s.buffer_type), size(s.tile_type), grid)
        bd = IR.Block(IR.Type[], IR.Location[])
        push!(bd, dma_bd_op(ctx, args[i], dims, len; offset))
        push!(bd, end_op(ctx))
        t = dma_configure_task_for_op(ctx, s.name, region(bd); issue_token = token)
        push!(body, t)
        push!(body, dma_start_task_op(ctx, IR.result(t, 1)))
        return IR.result(t, 1)
    end

    grid_of(s, coord) = Tuple(coord[a] for a in s.access)

    for sc in _axis_coords(space)
        pending, outs = IR.Value[], IR.Value[]
        for rc in _axis_coords(reduction)
            full = merge(sc, rc)
            for (i, s) in enumerate(specs)
                s.dir === :in || continue
                push!(pending, task(i, grid_of(s, full); token = false))
            end
        end
        for (i, s) in enumerate(specs)
            s.dir === :out || continue
            push!(outs, task(i, grid_of(s, sc); token = true))
        end
        for o in outs
            push!(body, dma_await_task_op(ctx, o))
        end
        for p in pending
            push!(body, dma_free_task_op(ctx, p))
        end
    end
    return runtime_sequence_op(ctx, "sequence", region(body))
end

function _build_schedule_program(
        @nospecialize(init), @nospecialize(step), specs, space, reduction,
        device::AIEDevice, name::AbstractString, stack_size::Int; ctx::IR.Context = context(),
    )
    num_space = isempty(space) ? 1 : prod(Int(e) for (_, e) in space)
    num_reduce = isempty(reduction) ? 1 : prod(Int(e) for (_, e) in reduction)

    device_body = IR.Block(IR.Type[], IR.Location[])
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
        push!(device_body, objectfifo_op(ctx, s.name, producer_tile, IR.Value[consumer_tile], objectfifo_type(ctx, s.tile_type), 2))
    end
    push!(device_body, core_op(ctx, core_tile, _emit_schedule_core!(ctx, init, step, specs, num_space, num_reduce); stack_size))
    push!(device_body, _emit_schedule_runtime!(ctx, specs, space, reduction))
    push!(device_body, end_op(ctx))

    mod = IR.Module(loc(ctx))
    push!(IR.body(mod), device_op(ctx, device, name, region(device_body)))
    IR.verify(IR.Operation(mod)) ||
        error("IRON: @iron_schedule generated an invalid MLIR module (see the diagnostics above)")
    canonicalize!(mod, ctx)
    return string(IR.Operation(mod))
end

# --- macro front end ---------------------------------------------------------

# `(mi = M÷m, nj = N÷n)` or a single `(kk = K÷k)` -> [(:mi, :(M÷m)), ...].
function _parse_axes(rhs)
    axes = Tuple{Symbol, Any}[]
    entries = Meta.isexpr(rhs, :tuple) ? rhs.args : Any[rhs]
    for e in entries
        (Meta.isexpr(e, :(=)) && e.args[1] isa Symbol) ||
            error("@iron_schedule: an axis must be written `name = extent`, got `$e`")
        push!(axes, (e.args[1], e.args[2]))
    end
    return axes
end

# `(mi, kk)` or a single `mi` -> [:mi, :kk].
function _parse_access(a)
    a isa Symbol && return Symbol[a]
    (Meta.isexpr(a, :tuple) && all(x -> x isa Symbol, a.args)) ||
        error("@iron_schedule: an access must be `(axis, ...)` of axis names, got `$a`")
    return Symbol[a.args...]
end

_axes_to_expr(axes) = Expr(:tuple, [Expr(:tuple, QuoteNode(nm), esc(ex)) for (nm, ex) in axes]...)

"""
    @iron_schedule begin
        space  = (mi = M÷m, nj = N÷n)
        reduce = (kk = K÷k,)
        init   = zero_kernel
        step   = acc_kernel
        In(A)  => (mi, kk)
        In(B)  => (kk, nj)
        Out(C) => (mi, nj)
    end

Compile and run a tiled **reduction** on the NPU -- the schedule `@iron` cannot infer.
The design accumulates an output tile across an inner loop that streams the input
operands and reduces into it, GEMM being the archetype.

The block declares:

  * `space` -- the output-tile iteration axes as `name = extent` (the outer loop nest).
  * `reduce` -- the accumulation axes (the inner loop nest); omit for none.
  * `init` -- a kernel run on the output/accumulator tiles once per output tile, taking
    the `Out` operands in declaration order (e.g. `zero_kernel(c)`).
  * `step` -- a kernel run each reduction step, taking **all** operands in declaration
    order (e.g. `acc_kernel(a, b, c)`), accumulating into the outputs.
  * one line per operand, `In(a) => (axes...)` or `Out(a) => (axes...)`, giving the
    direction and the axes indexing that buffer, one per dimension. Each `a` is an
    [`NPUArray`](@ref); the tile shape is inferred as `buffer_dim ÷ axis_extent` per
    dimension, so the axis extents must divide the buffer. An `Out` operand is the
    accumulator and may be indexed by space axes only.

Options may be added as block settings: `device`, `name`, `flags`, `verbose`,
`stack_size` (reductions often need a larger core stack than the 1024-byte default).

The compiled design is cached like [`@iron`](@ref). Results land in the `Out`
buffers, ready via `Array`. Returns the [`CompiledProgram`](@ref).

```julia
# C = A * B, in (m, k) x (k, n) tiles reduced on one core.
@iron_schedule begin
    space  = (mi = M÷m, nj = N÷n)
    reduce = (kk = K÷k,)
    init   = gemm_zero!
    step   = gemm_acc!
    stack_size = 3328
    flags  = ["--alloc-scheme=basic-sequential"]
    In(A)  => (mi, kk)
    In(B)  => (kk, nj)
    Out(C) => (mi, nj)
end
```
"""
macro iron_schedule(block)
    Meta.isexpr(block, :block) ||
        error("@iron_schedule: expected a `begin ... end` block")

    space = reduction = init = step = nothing
    operands = Tuple{Symbol, Any, Vector{Symbol}}[]
    options = Expr[]
    for stmt in block.args
        stmt isa LineNumberNode && continue
        if Meta.isexpr(stmt, :(=))
            lhs, rhs = stmt.args
            if lhs === :space
                space = _parse_axes(rhs)
            elseif lhs === :reduce
                reduction = _parse_axes(rhs)
            elseif lhs === :init
                init = rhs
            elseif lhs === :step
                step = rhs
            elseif lhs isa Symbol
                push!(options, Expr(:kw, lhs, esc(rhs)))
            else
                error("@iron_schedule: unexpected setting `$stmt`")
            end
        elseif Meta.isexpr(stmt, :call) && length(stmt.args) == 3 && stmt.args[1] === :(=>)
            lhsop, access = stmt.args[2], stmt.args[3]
            (Meta.isexpr(lhsop, :call) && lhsop.args[1] in (:In, :Out) && length(lhsop.args) == 2) ||
                error("@iron_schedule: an operand must be `In(a) => (axes...)` or `Out(a) => (axes...)`, got `$stmt`")
            dir = lhsop.args[1] === :In ? :in : :out
            push!(operands, (dir, lhsop.args[2], _parse_access(access)))
        else
            error("@iron_schedule: unexpected line `$stmt`")
        end
    end

    space === nothing && error("@iron_schedule: missing `space = (...)`")
    init === nothing && error("@iron_schedule: missing `init = kernel`")
    step === nothing && error("@iron_schedule: missing `step = kernel`")
    isempty(operands) && error("@iron_schedule: no operands declared")
    reduction === nothing && (reduction = Tuple{Symbol, Any}[])

    ops_expr = Expr(:tuple, [
        Expr(:tuple, QuoteNode(dir), esc(arr), Expr(:tuple, QuoteNode.(access)...))
        for (dir, arr, access) in operands
    ]...)

    return Expr(
        :call, _schedule_launch,
        _axes_to_expr(space), _axes_to_expr(reduction), esc(init), esc(step), ops_expr,
        options...,
    )
end
