# Compiles a Julia function into MLIR ops emitted directly into a caller-provided
# block. IRStructurizer recovers scf-style structured control flow from Julia's
# IR; this file translates the result op-by-op.

# Julia's `Int` becomes `index` so it can drive loop bounds and memref subscripts;
# sized integers and floats map to their MLIR equivalents.
function mlir_type(ctx::IR.Context, T)
    T === Int && return IR.IndexType(; context = ctx)
    T <: Tile && return memref_type(ctx, T)
    return IR.Type(T; context = ctx)
end

# Julia intrinsic => (MLIR builder, cmpi predicate or nothing).
# arith.cmpi predicates: 0 eq, 1 ne, 2 slt, 3 sle, 4 sgt, 5 sge.
const ARITH_OPS = Dict{Any, Tuple{Any, Union{Nothing, Int}}}(
    Base.add_int => (arith.addi, nothing),
    Base.sub_int => (arith.subi, nothing),
    Base.mul_int => (arith.muli, nothing),
    Base.sdiv_int => (arith.divsi, nothing),
    Base.srem_int => (arith.remsi, nothing),
    Base.checked_srem_int => (arith.remsi, nothing),
    Base.and_int => (arith.andi, nothing),
    Base.or_int => (arith.ori, nothing),
    Base.xor_int => (arith.xori, nothing),
    Base.add_float => (arith.addf, nothing),
    Base.sub_float => (arith.subf, nothing),
    Base.mul_float => (arith.mulf, nothing),
    Base.div_float => (arith.divf, nothing),
    Base.slt_int => (arith.cmpi, 2),
    Base.sle_int => (arith.cmpi, 3),
    Base.eq_int => (arith.cmpi, 0),
    Base.ne_int => (arith.cmpi, 1),
    Base.:(===) => (arith.cmpi, 0),
)

"""
    KernelContext

Threads the MLIR context and the Julia-value => MLIR-value map through emission.
`values` maps IRStructurizer SSA values, block arguments and `Core.Argument`s to
MLIR `Value`s -- except for `ForOp`/`IfOp`, which map to the `Operation` itself so
that a downstream `getfield` on their tuple result can select the right result.
`constants` memoizes materialized literals, keyed by type and value, and `indices`
memoizes 0-based subscripts. `used` is the set of SSA values the kernel actually
reads, used to skip emitting dead arithmetic.
"""
struct KernelContext
    ctx::IR.Context
    values::Dict{Any, Any}
    constants::Dict{Tuple{DataType, Any}, IR.Value}
    indices::Dict{Any, IR.Value}
    used::IRStructurizer.UseIndex
end

KernelContext(ctx::IR.Context, used::IRStructurizer.UseIndex) = KernelContext(
    ctx, Dict{Any, Any}(), Dict{Tuple{DataType, Any}, IR.Value}(), Dict{Any, IR.Value}(), used
)

# Copied when descending into a nested region: values defined in the enclosing block
# dominate the nested one and stay visible, while anything the nested region defines
# is confined to its copy.
Base.copy(kc::KernelContext) = KernelContext(
    kc.ctx, copy(kc.values), copy(kc.constants), copy(kc.indices), kc.used
)

# Resolve a Julia value to an MLIR Value, materializing an arith.constant for literals.
#
# Constants are cached per (type, value): keying on the value alone would conflate
# literals that Julia considers equal but MLIR does not, so that `Int32(1)` and the
# `index` 1 would share one constant and produce a type-mismatched op.
function lookup!(kc::KernelContext, block::IR.Block, v)
    haskey(kc.values, v) && return kc.values[v]
    v isa Union{Integer, AbstractFloat, Bool} ||
        error("IRON: cannot materialize $(repr(v))::$(typeof(v)) as an MLIR value")
    key = (typeof(v), v)
    haskey(kc.constants, key) && return kc.constants[key]
    type = mlir_type(kc.ctx, typeof(v))
    op = arith.constant(; value = IR.Attribute(v, type), location = loc(kc.ctx))
    push!(block, op)
    result = IR.result(op, 1)
    kc.constants[key] = result
    return result
end

# Tiles are indexed from 1 like any Julia array, memrefs from 0, so every subscript
# loses one on the way down. A literal index folds here; a computed one costs one
# arith.subi, memoized so that `a[i]` and `b[i]` share it.
function memref_index!(kc::KernelContext, block::IR.Block, index)
    index isa Integer && return lookup!(kc, block, Int(index) - 1)
    haskey(kc.indices, index) && return kc.indices[index]
    value = lookup!(kc, block, index)
    one_ = lookup!(kc, block, 1)
    op = arith.subi(value, one_; result = IR.IndexType(; context = kc.ctx), location = loc(kc.ctx))
    push!(block, op)
    result = IR.result(op, 1)
    kc.indices[index] = result
    return result
end

function emit_call!(kc::KernelContext, block::IR.Block, jblock, inst)
    resolved = IRStructurizer.resolve_call(jblock, inst)
    resolved === nothing && return nothing
    fn, ops = resolved
    ssa = IRStructurizer.SSAValue(inst[:ssa_idx])

    # A ForOp/IfOp result is a Julia tuple; `getfield(op, i)` selects one of the
    # MLIR op's results. `kc.values` holds the Operation for exactly this case.
    if fn === Core.getfield || fn === Base.getfield
        source = kc.values[ops[1]]
        kc.values[ssa] = IR.result(source, ops[2])
        return nothing
    end

    if fn === Base.getindex
        tile = lookup!(kc, block, ops[1])
        index = memref_index!(kc, block, ops[2])
        op = memref.load(
            tile, IR.Value[index];
            result = mlir_type(kc.ctx, inst[:type]), location = loc(kc.ctx),
        )
        push!(block, op)
        kc.values[ssa] = IR.result(op, 1)
        return nothing
    end

    if fn === Base.setindex!
        tile = lookup!(kc, block, ops[1])
        value = lookup!(kc, block, ops[2])
        index = memref_index!(kc, block, ops[3])
        push!(block, memref.store(value, tile, IR.Value[index]; location = loc(kc.ctx)))
        return nothing
    end

    entry = get(ARITH_OPS, fn, nothing)
    entry === nothing && error("IRON: no lowering registered for $fn")
    builder, predicate = entry

    # Arithmetic is pure, so an unused result means the whole op is dead. These do
    # show up: structurizing a loop turns its exit test into `scf.for` bounds and
    # leaves the original comparison behind with no remaining reader.
    haskey(kc.used, ssa) || return nothing

    args = IR.Value[lookup!(kc, block, o) for o in ops]
    result = mlir_type(kc.ctx, inst[:type])
    op = if predicate === nothing
        builder(args...; result, location = loc(kc.ctx))
    else
        builder(
            args...; result,
            predicate = IR.Attribute(predicate, IR.Type(Int64; context = kc.ctx)),
            location = loc(kc.ctx),
        )
    end
    push!(block, op)
    kc.values[ssa] = IR.result(op, 1)
    return nothing
end

function emit_block!(kc::KernelContext, block::IR.Block, jblock)
    for inst in IRStructurizer.instructions(jblock)
        stmt = inst[:stmt]
        if stmt isa IRStructurizer.ForOp
            emit_for!(kc, block, stmt, inst)
        elseif stmt isa IRStructurizer.IfOp
            emit_if!(kc, block, stmt, inst)
        elseif IRStructurizer.iscall(inst)
            emit_call!(kc, block, jblock, inst)
        end
    end
    return nothing
end

# Build an MLIR block for a structured region's body. `extra_args` seats leading
# block arguments that IRStructurizer does not report in `arguments`, namely a
# ForOp's induction variable.
function build_body!(kc::KernelContext, jbody; extra_args = Pair[])
    body_args = IRStructurizer.arguments(jbody)
    arg_types = IR.Type[
        IR.Type[t for (_, t) in extra_args];
        [mlir_type(kc.ctx, a.type) for a in body_args]
    ]
    block = IR.Block(arg_types, [loc(kc.ctx) for _ in arg_types])

    inner = copy(kc)
    for (i, (jval, _)) in enumerate(extra_args)
        inner.values[jval] = IR.argument(block, i)
    end
    for (i, a) in enumerate(body_args)
        inner.values[a] = IR.argument(block, length(extra_args) + i)
    end

    emit_block!(inner, block, jbody)
    return block, inner
end

# Emit the scf.yield closing a structured region, returning the yielded values.
function emit_yield!(kc::KernelContext, block::IR.Block, jbody)
    term = IRStructurizer.terminator(jbody)
    values = IR.Value[lookup!(kc, block, v) for v in IRStructurizer.operands(term)]
    push!(block, scf.yield(values; location = loc(kc.ctx)))
    return values
end

function emit_for!(kc::KernelContext, block::IR.Block, forop, inst)
    lower = lookup!(kc, block, forop.lower)
    upper = lookup!(kc, block, forop.upper)
    step = lookup!(kc, block, forop.step)
    inits = IR.Value[lookup!(kc, block, v) for v in forop.init_values]

    body, inner = build_body!(
        kc, forop.body;
        extra_args = [forop.iv_arg => IR.IndexType(; context = kc.ctx)],
    )
    emit_yield!(inner, body, forop.body)

    op = scf.for_(
        lower, upper, step, inits;
        region = region(body), results = [IR.type(v) for v in inits], location = loc(kc.ctx),
    )
    push!(block, op)
    kc.values[IRStructurizer.SSAValue(inst[:ssa_idx])] = op
    return nothing
end

function emit_if!(kc::KernelContext, block::IR.Block, ifop, inst)
    (cond,) = IRStructurizer.operands(ifop)
    cond_value = lookup!(kc, block, cond)
    jblocks = IRStructurizer.blocks(ifop)

    then_block, then_kc = build_body!(kc, jblocks[1])
    then_yields = emit_yield!(then_kc, then_block, jblocks[1])
    then_region = region(then_block)

    else_region = IR.Region()
    if length(jblocks) > 1
        else_block, else_kc = build_body!(kc, jblocks[2])
        emit_yield!(else_kc, else_block, jblocks[2])
        push!(else_region, else_block)
    end

    op = scf.if_(
        cond_value;
        thenRegion = then_region, elseRegion = else_region,
        results = [IR.type(y) for y in then_yields], location = loc(kc.ctx),
    )
    push!(block, op)
    kc.values[IRStructurizer.SSAValue(inst[:ssa_idx])] = op
    return nothing
end

"""
    compile_kernel!(ctx, block, f, argtypes, args)

Compile Julia function `f`, called with `argtypes`, and emit its body into `block`.
`args` are the MLIR values bound to `f`'s parameters, so the kernel is inlined at
the point of use rather than emitted as a separate function.

Only kernels returning `nothing` are supported: a kernel communicates through the
tiles it writes.
"""
function compile_kernel!(
        ctx::IR.Context, block::IR.Block, @nospecialize(f), @nospecialize(argtypes),
        args::Vector{IR.Value},
    )
    results = IRStructurizer.code_structured(f, argtypes)
    length(results) == 1 ||
        error("IRON: expected exactly one method of $f for $argtypes, found $(length(results))")
    sci, rettype = only(results)
    rettype === Nothing ||
        error("IRON: kernel $f must return nothing, got $rettype")

    entry = first(eachblock(sci))
    kc = KernelContext(ctx, IRStructurizer.uses(entry))
    # Parameters are referenced as Core.Argument(i+1); Argument(1) is #self#.
    for (i, arg) in enumerate(args)
        kc.values[Core.Argument(i + 1)] = arg
    end
    emit_block!(kc, block, entry)
    return nothing
end
