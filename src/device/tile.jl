# The array type a kernel sees, plus the intrinsics the kernel compiler maps onto
# memref operations.

"""
    Tile{T,Dims}

A tile of device memory, as seen from inside a kernel. `T` is the element type and
`Dims` is a `Tuple` type of the extents, e.g. `Tile{Int32,Tuple{1024}}` lowers to
`memref<1024xi32>` and `Tile{Float32,Tuple{8,16}}` to `memref<8x16xf32>`.

A `Tile` has no host representation and is never constructed: it exists so that a
kernel can be type-inferred against it, and its `getindex`/`setindex!` are markers
that the kernel compiler rewrites into `memref.load`/`memref.store`.

Indexing is 1-based as everywhere else in Julia, and takes one subscript per
dimension; the compiler subtracts the offset when lowering to the underlying
0-based memref.
"""
struct Tile{T, Dims} end

Base.eltype(::Type{Tile{T, Dims}}) where {T, Dims} = T
Base.size(::Type{Tile{T, Dims}}) where {T, Dims} = tuple(Dims.parameters...)
Base.ndims(::Type{Tile{T, Dims}}) where {T, Dims} = length(Dims.parameters)
Base.length(T::Type{<:Tile}) = prod(size(T))

# These two bodies are never executed and never lowered; only the call itself is
# meaningful. But they must be opaque to inference: a body that returns a literal
# lets inference conclude the load is `Const(0)` and constant-fold the arithmetic
# that consumes it, silently erasing the kernel. `inferencebarrier` keeps the
# result typed as `T` while blocking constant propagation, and `@noinline` keeps
# the call visible in the IR instead of being inlined into its (meaningless) body.

@noinline function Base.getindex(tile::Tile{T}, I::Int...) where {T}
    return Base.inferencebarrier(zero(T))::T
end

@noinline function Base.setindex!(tile::Tile{T}, v::T, I::Int...) where {T}
    Base.donotdelete(tile, v, I)
    return nothing
end

"""
    mlir_eltype(ctx, T) -> IR.Type

The MLIR type that a tile element of Julia type `T` lowers to.

Overload this for element types MLIR.jl does not already map, which is how the
FP8 formats are attached -- see `ext/`.
"""
mlir_eltype(ctx::IR.Context, ::Type{T}) where {T} = IR.Type(T; context = ctx)

"""
    bitwidth(T) -> Int

Width of `T` in bits, used to pick between widening and narrowing conversions.

The default assumes `T` occupies whole bytes. Overload it for sub-byte formats,
whose `sizeof` over-counts because they are stored one per byte.
"""
bitwidth(::Type{T}) where {T} = 8 * sizeof(T)

"""
    memref_type(ctx, ::Type{Tile{T,Dims}}) -> IR.Type

The `memref` type a `Tile` lowers to.
"""
function memref_type(ctx::IR.Context, ::Type{Tile{T, Dims}}) where {T, Dims}
    dims = Int[Dims.parameters...]
    element = mlir_eltype(ctx, T)
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
