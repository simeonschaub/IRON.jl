using IRStructurizer
using MLIR: IR, API
using MLIR.Dialects: scf, arith, func

IR.mark_donate(op) = (IR.mark_dispose(op); op)
IRStructurizer.operands(::IRStructurizer.Block, r::Core.ReturnNode) = Any[r.val]

function loopsum(n::Int)
    s = 0
    i = 0
    while i < n
        if Base.srem_int(i, 2) == 0
            s += i
        end
        i += 1
    end
    return s
end

# --- inspect the structurized IR first, so you can see what you're lowering ---
sci, rettype = only(IRStructurizer.code_structured(loopsum, Tuple{Int}))
println(sci)

# --- dispatch table: Julia intrinsic -> MLIR op constructor ---
# names taken directly from the printed IR (checked_srem_int, ===), not the generic ones
const ARITH_TABLE = Dict(
    Base.add_int         => (arith.addi, nothing),
    Base.sub_int          => (arith.subi, nothing),
    Base.mul_int          => (arith.muli, nothing),
    Base.checked_srem_int => (arith.remsi, nothing),
    Base.srem_int         => (arith.remsi, nothing),
    Base.slt_int          => (arith.cmpi, 2), # :slt),
    Base.sle_int          => (arith.cmpi, 3), # :sle),
    Base.eq_int           => (arith.cmpi, 0), # :eq),
    Base.:(===)           => (arith.cmpi, 0), # :eq),   # Julia lowers `x == literal` to === for Int
)

function lookup!(ctx, block, valmap, v)
    haskey(valmap, v) && return valmap[v]
    c = arith.constant(; value = IR.Attribute(v, mlir_type(ctx, typeof(v))), location = IR.Location(; context = ctx))
    push!(block, c)
    r = IR.result(c, 1)
    valmap[v] = r
    return r
end

function emit_call!(ctx, block, jblock, valmap, inst)
    fn, ops = IRStructurizer.resolve_call(jblock, inst)

    # special case: getfield on a tuple-wrapped ForOp/IfOp result -> index into the
    # real MLIR op's results directly. valmap[obj] holds the *Operation*, not a Value,
    # for exactly this reason -- see emit_for!/emit_if! below.
    if fn === Core.getfield || fn === Base.getfield
        obj, idx = ops[1], ops[2]
        srcop = valmap[obj]
        valmap[IRStructurizer.SSAValue(inst[:ssa_idx])] = IR.result(srcop, idx)
        return
    end

    entry = get(ARITH_TABLE, fn, nothing)
    entry === nothing && error("no lowering registered for $fn")
    ctor, pred = entry
    args = [lookup!(ctx, block, valmap, o) for o in ops]
    loc = IR.Location(; context = ctx)
    op = pred === nothing ? ctor(args...; result = mlir_type(ctx, inst[:type]), location = loc) :
                            ctor(args...; result = mlir_type(ctx, inst[:type]), predicate = IR.Attribute(pred, IR.Type(Int; context = ctx)), location = loc)
    push!(block, op)
    valmap[IRStructurizer.SSAValue(inst[:ssa_idx])] = IR.result(op, 1)
end

function emit_region!(ctx, block, jblock, valmap)
    for inst in IRStructurizer.instructions(jblock)
        stmt = inst[:stmt]
        if stmt isa IRStructurizer.ForOp
            emit_for!(ctx, block, stmt, valmap, inst)
        elseif stmt isa IRStructurizer.IfOp
            emit_if!(ctx, block, stmt, valmap, inst)
        elseif IRStructurizer.iscall(inst)
            emit_call!(ctx, block, jblock, valmap, inst)
        end
    end
end

function mlir_type(ctx, julia_type)
    julia_type == Int && return IR.IndexType(; context = ctx)
    return IR.Type(julia_type; context = ctx)
end

# extra_args: Vector{Pair} of (julia_value => mlir_type) to prepend as leading block
# args before the block's own `arguments(jbody)` -- used to seat ForOp's iv_arg, which
# (confirmed earlier) is NOT part of arguments(body).
function build_body_block(ctx, jbody, valmap; extra_args = Pair[])
    body_args = IRStructurizer.arguments(jbody)
    extra_types = IR.Type[t for (_, t) in extra_args]
    arg_types = IR.Type[extra_types; [mlir_type(ctx, a.type) for a in body_args]]
    locs = [IR.Location(; context = ctx) for _ in arg_types]
    mlir_block = IR.Block(arg_types, locs)

    inner_valmap = copy(valmap)
    offset = length(extra_args)
    for (i, (jval, _)) in enumerate(extra_args)
        inner_valmap[jval] = IR.argument(mlir_block, i)
    end
    for (i, a) in enumerate(body_args)
        inner_valmap[a] = IR.argument(mlir_block, offset + i)
    end

    emit_region!(ctx, mlir_block, jbody, inner_valmap)
    return mlir_block, inner_valmap
end

function emit_for!(ctx, block, forop, valmap, inst)
    # use the real fields, confirmed via fieldnames(): (:lower, :upper, :step, :iv_arg, :body, :init_values)
    lower_v = lookup!(ctx, block, valmap, forop.lower)
    upper_v = lookup!(ctx, block, valmap, forop.upper)
    step_v  = lookup!(ctx, block, valmap, forop.step)
    init_vs = [lookup!(ctx, block, valmap, v) for v in forop.init_values]

    jbody = forop.body
    mlir_body, inner_valmap = build_body_block(
        ctx, jbody, valmap;
        extra_args = [forop.iv_arg => IR.IndexType(; context = ctx)],
    )

    term = IRStructurizer.terminator(jbody)   # a ContinueOp
    yield_vals = [lookup!(ctx, mlir_body, inner_valmap, v) for v in IRStructurizer.operands(term)]
    push!(mlir_body, scf.yield(yield_vals; location = IR.Location(; context = ctx)))

    region = IR.Region()
    push!(region, mlir_body)

    results = [IR.type(v) for v in init_vs]
    op = scf.for_(lower_v, upper_v, step_v, init_vs; region, results, location = IR.Location(; context = ctx))
    push!(block, op)

    # store the whole Operation, not a single result -- ForOp's Julia-side result is a
    # Tuple that downstream code unwraps via getfield(%16, i); emit_call!'s getfield
    # special-case does that unwrapping against this Operation.
    valmap[IRStructurizer.SSAValue(inst[:ssa_idx])] = op
end

function emit_if!(ctx, block, ifop, valmap, inst)
    (cond,) = IRStructurizer.operands(ifop)
    cond_v = lookup!(ctx, block, valmap, cond)

    subblocks = IRStructurizer.blocks(ifop)
    then_jblock = subblocks[1]
    then_block, then_valmap = build_body_block(ctx, then_jblock, valmap)
    then_term = IRStructurizer.terminator(then_jblock)
    then_yields = [lookup!(ctx, then_block, then_valmap, v) for v in IRStructurizer.operands(then_term)]
    push!(then_block, scf.yield(then_yields; location = IR.Location(; context = ctx)))
    thenRegion = IR.Region(); push!(thenRegion, then_block)

    elseRegion = IR.Region()
    if length(subblocks) > 1
        else_jblock = subblocks[2]
        else_block, else_valmap = build_body_block(ctx, else_jblock, valmap)
        else_term = IRStructurizer.terminator(else_jblock)
        else_yields = [lookup!(ctx, else_block, else_valmap, v) for v in IRStructurizer.operands(else_term)]
        push!(else_block, scf.yield(else_yields; location = IR.Location(; context = ctx)))
        push!(elseRegion, else_block)
    end

    results = [IR.type(y) for y in then_yields]
    op = scf.if_(cond_v; thenRegion, elseRegion, results, location = IR.Location(; context = ctx))
    push!(block, op)

    # same reasoning as ForOp: store the Operation so getfield(%14, i) resolves correctly
    valmap[IRStructurizer.SSAValue(inst[:ssa_idx])] = op
end

# --- top-level driver ---
function lower_to_mlir(f, argtypes)
    registry = IR.DialectRegistry()
    API.mlirRegisterAllDialects(registry)
    ctx = IR.Context(registry)
    IR.get_or_load_dialect!("func"; context = ctx)
    IR.get_or_load_dialect!("scf"; context = ctx)
    IR.get_or_load_dialect!("arith"; context = ctx)

    sci, rettype = only(IRStructurizer.code_structured(f, argtypes))
    entry = first(eachblock(sci))

    arg_types = [mlir_type(ctx, t) for t in argtypes.parameters]
    fn_type = IR.FunctionType(arg_types, [mlir_type(ctx, rettype)]; context = ctx)
    fn_block = IR.Block(arg_types, [IR.Location(; context = ctx) for _ in arg_types])

    valmap = Dict{Any,Any}()
    # function parameters are Core.Argument nodes referenced directly in the body
    # (printed as `_2`, `_3`, ...), NOT arguments(entry) -- entry has no block args of
    # its own. Core.Argument(1) is #self# and has no MLIR counterpart.
    for (i, _) in enumerate(argtypes.parameters)
        valmap[Core.Argument(i + 1)] = IR.argument(fn_block, i)
    end

    emit_region!(ctx, fn_block, entry, valmap)

    term = IRStructurizer.terminator(entry)
    ret_vals = [lookup!(ctx, fn_block, valmap, v) for v in IRStructurizer.operands(entry, term)]
    push!(fn_block, func.return_(ret_vals; location = IR.Location(; context = ctx)))

    fn_region = IR.Region(); push!(fn_region, fn_block)
    fn_op = func.func_(; sym_name = IR.Attribute("loopsum"; context = ctx), function_type = fn_type,
                          body = fn_region, location = IR.Location(; context = ctx))

    mod = IR.Module(IR.Location(; context = ctx))
    push!(IR.body(mod), fn_op)
    return mod
end

mod = lower_to_mlir(loopsum, Tuple{Int})
println(mod)
