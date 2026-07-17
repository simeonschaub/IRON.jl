# The array type a kernel sees, plus the intrinsics the kernel compiler maps onto
# memref operations.

"""
    Tile{T,Dims}

A tile of device memory, as seen from inside a kernel. `T` is the element type and
`Dims` is a `Tuple` type of the extents, e.g. `Tile{Int32,Tuple{1024}}` lowers to
`memref<1024xi32>`.

A `Tile` has no host representation and is never constructed: it exists so that a
kernel can be type-inferred against it, and its `getindex`/`setindex!` are markers
that the kernel compiler rewrites into `memref.load`/`memref.store`.

Indexing is 1-based as everywhere else in Julia; the compiler subtracts the offset
when lowering to the underlying 0-based memref.
"""
struct Tile{T, Dims} end

Base.eltype(::Type{Tile{T, Dims}}) where {T, Dims} = T
Base.size(::Type{Tile{T, Dims}}) where {T, Dims} = tuple(Dims.parameters...)
Base.length(::Type{<:Tile}) = prod(size(Tile))

# These two bodies are never executed and never lowered; only the call itself is
# meaningful. But they must be opaque to inference: a body that returns a literal
# lets inference conclude the load is `Const(0)` and constant-fold the arithmetic
# that consumes it, silently erasing the kernel. `inferencebarrier` keeps the
# result typed as `T` while blocking constant propagation, and `@noinline` keeps
# the call visible in the IR instead of being inlined into its (meaningless) body.

@noinline function Base.getindex(tile::Tile{T}, i::Int) where {T}
    return Base.inferencebarrier(zero(T))::T
end

@noinline function Base.setindex!(tile::Tile{T}, v::T, i::Int) where {T}
    Base.donotdelete(tile, v, i)
    return nothing
end

"""
    memref_type(ctx, ::Type{Tile{T,Dims}}) -> IR.Type

The `memref` type a `Tile` lowers to.
"""
function memref_type(ctx::IR.Context, ::Type{Tile{T, Dims}}) where {T, Dims}
    dims = Int[Dims.parameters...]
    element = IR.Type(T; context = ctx)
    return IR.Type(API.mlirMemRefTypeContiguousGet(element, length(dims), dims, IR.Attribute()))
end

"""
    objectfifo_type(ctx, ::Type{<:Tile}) -> IR.Type

The `!aie.objectfifo<memref<...>>` type for a FIFO carrying this tile.
"""
objectfifo_type(ctx::IR.Context, T::Type{<:Tile}) =
    opaque_type("!aie.objectfifo<$(memref_type(ctx, T))>"; context = ctx)

"""
    objectfifo_subview_type(ctx, ::Type{<:Tile}) -> IR.Type

The `!aie.objectfifosubview<memref<...>>` type produced by acquiring from a FIFO.
"""
objectfifo_subview_type(ctx::IR.Context, T::Type{<:Tile}) =
    opaque_type("!aie.objectfifosubview<$(memref_type(ctx, T))>"; context = ctx)
