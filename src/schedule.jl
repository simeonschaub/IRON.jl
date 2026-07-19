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

# The launch behind the `for` form. `space`/`reduction` are tuples of `(name, extent)`;
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
        error("IRON: an @iron reduction needs at least one `Out(...)` operand (the accumulator)")
    any(s -> s.dir === :in, specs) ||
        error("IRON: an @iron reduction needs at least one `In(...)` operand")

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

# The core loop nest: acquire and initialise the outputs in the outer (space) loop,
# then acquire the inputs and run the step kernel in the inner (reduce) loop,
# accumulating into the outputs held across it. The core never
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

# The host DMA: for each output tile (a space coordinate), stream every input tile the
# reduction reads -- addressing each operand
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
    for stmt in block.args
        stmt isa LineNumberNode && continue
        if Meta.isexpr(stmt, :macrocall) && stmt.args[1] === Symbol("@init")
            init === nothing || error("@iron: more than one `@init` in the `for` body")
            initexpr = stmt.args[end]
            init = Meta.isexpr(initexpr, :call) ? initexpr.args[1] : initexpr
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

    ops_expr = Expr(:tuple, [
        Expr(:tuple, QuoteNode(dir), esc(arr), Expr(:tuple, QuoteNode.(access)...))
        for (dir, arr, access) in operands
    ]...)

    return Expr(
        :call, _schedule_launch,
        _axes_to_expr(space), _axes_to_expr(reduction),
        init === nothing ? :nothing : esc(init), esc(step), ops_expr,
        kws...,
    )
end
