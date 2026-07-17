# MLIR context setup and the generic-operation layer used to build `aie`/`aiex` ops.
#
# MLIR.jl links against upstream MLIR, which does not know about the out-of-tree
# `aie`/`aiex` dialects. Rather than build Julia bindings for them, we enable
# unregistered dialects and construct their ops generically. MLIR keeps an opaque
# properties slot on unregistered ops, so `<{...}>` inherent attributes and
# `#aie<...>` / `!aie.<...>` attributes and types all round-trip through the
# textual form. `aie-opt` then parses that text with the real dialects registered.

"""
    context() -> IR.Context

An MLIR context with the upstream dialects registered and unregistered dialects
allowed, so that `aie`/`aiex` operations can be built generically.
"""
function context()
    registry = IR.DialectRegistry()
    API.mlirRegisterAllDialects(registry)
    API.mlirRegisterAllPasses()
    ctx = IR.Context(registry)
    IR.allow_unregistered_dialects!(true; context = ctx)
    for dialect in ("func", "scf", "arith", "memref", "vector")
        IR.get_or_load_dialect!(dialect; context = ctx)
    end
    return ctx
end

"""
    canonicalize!(mod, ctx) -> mod

Run `canonicalize` and `cse` over `mod`.

Structurizing Julia's IR leaves a loop carrying values it never changes -- the
enclosing loops' induction variables, threaded through as if they were
accumulators. Canonicalization drops them, which brings the core body down to the
one carried value the kernel actually has and folds the constant loop bounds. The
`aie` ops pass through untouched: they are unregistered here, so they carry no
patterns, and an unregistered op is assumed to have side effects and is not
removed.
"""
function canonicalize!(mod::IR.Module, ctx::IR.Context)
    pm = IR.PassManager(; context = ctx)
    # The pipeline is added to the top-level manager unnested. Wrapping it in
    # `builtin.module(...)` would build a manager anchored at *contained* modules,
    # of which there are none, and every pass would silently not run.
    IR.add_pipeline!(IR.OpPassManager(pm), "canonicalize,cse")
    IR.run!(pm, mod)
    return mod
end

"""
    opaque_attr(str; context) -> IR.Attribute

Parse an attribute belonging to an unregistered dialect, e.g.
`#aie<bd_dim_layout_array[]>`. It becomes an `OpaqueAttr` carrying the verbatim
text and prints back identically.
"""
opaque_attr(str::AbstractString; context::IR.Context) =
    IR.Attribute(API.mlirAttributeParseGet(context, String(str)))

"""
    opaque_type(str; context) -> IR.Type

Parse a type belonging to an unregistered dialect, e.g.
`!aie.objectfifo<memref<1024xi32>>`.
"""
opaque_type(str::AbstractString; context::IR.Context) =
    IR.Type(API.mlirTypeParseGet(context, String(str)))

"""
    region(block) -> IR.Region

A fresh region owning the single `block`.
"""
function region(block::IR.Block)
    r = IR.Region()
    push!(r, block)
    return r
end

"""
    create_op(name, location; operands, results, regions, properties, attributes)

Build an operation generically. `properties` are set as *inherent* attributes and
print inside `<{...}>`; `attributes` are discardable and print inside `{...}`.
Getting this split right matters: an inherent attribute placed in the discardable
dictionary is not moved into the op's properties when `aie-opt` re-parses the
generic form, and the op then fails verification with the attribute reported missing.
"""
function create_op(
        name::AbstractString,
        location::IR.Location;
        operands::Vector{IR.Value} = IR.Value[],
        results::Vector{IR.Type} = IR.Type[],
        regions::Vector{IR.Region} = IR.Region[],
        properties = Pair{String, IR.Attribute}[],
        attributes::Vector{IR.NamedAttribute} = IR.NamedAttribute[],
    )
    op = IR.create_operation_common(
        String(name),
        location;
        operands,
        results,
        owned_regions = regions,
        attributes,
        result_inference = false,
    )
    for (key, value) in properties
        API.mlirOperationSetInherentAttributeByName(op, String(key), value)
    end
    return op
end
