# The host-side array: an NPU-resident XRT buffer wearing the JuliaGPU
# `AbstractGPUArray` interface, so IRON composes with the rest of the ecosystem
# the way CUDA.jl, oneAPI.jl and Metal.jl do -- `Adapt` for converting a launch
# argument to its kernel-side view, and GPUArraysCore's scalar-indexing guard.
#
# This is the host counterpart to `Tile`, the *device*-side view a kernel sees.
# A host `NPUArray{T,N}` adapts to the kernel `Tile{T,Tuple{dims...}}`, exactly as
# a `CuArray` adapts to a `CuDeviceArray`. The buffer stays resident on the NPU
# (an XRT buffer object, allocated in `runtime.jl`); `NPUArray` is the typed Julia
# handle to it, its column-major storage mapped straight into host memory so the
# contents can be synced each way.

"""
    NPUArray{T,N} <: AbstractGPUArray{T,N}

An `N`-dimensional array of `T` living in NPU-resident memory, backed by an XRT
buffer object.

Construct one by copying a host array with `NPUArray(A)`, or allocate a zeroed
buffer of a given shape with `NPUArray{T}(undef, dims...)` (or, shaped like a
kernel tile, `NPUArray{T}(undef, Tile{T,Dims})`). Copy the contents back to the
host with `Array(x)`.

Pass `NPUArray`s to [`run!`](@ref) as the design's buffers. As a launch argument
each one adapts to the kernel's [`Tile`](@ref) view via `Adapt.adapt`.

Scalar indexing is disallowed by default -- touching one element still crosses to
the device and back to keep the host mapping coherent -- so wrap any deliberate
scalar access in `@allowscalar` (re-exported from GPUArraysCore).
"""
mutable struct NPUArray{T, N} <: AbstractGPUArray{T, N}
    bo::Ptr{Cvoid}      # XRT buffer object handle (owned; freed by finalizer)
    data::Array{T, N}   # host mapping of the buffer's memory (own = false)
end

function NPUArray(A::AbstractArray{T, N}) where {T, N}
    a = _npu_empty(T, size(A))
    copyto!(a.data, A)
    _xrt_bo_sync_to_device(a.bo)
    return a
end

NPUArray{T}(u::UndefInitializer, dims::Integer...) where {T} = NPUArray{T}(u, map(Int, dims))

function NPUArray{T}(::UndefInitializer, dims::Dims{N}) where {T, N}
    a = _npu_empty(T, dims)
    fill!(a.data, zero(T))
    _xrt_bo_sync_to_device(a.bo)
    return a
end

# Allocate a buffer shaped like a kernel tile, the common case for a design output.
NPUArray{T}(u::UndefInitializer, ::Type{Tile{T, Dims}}) where {T, Dims} =
    NPUArray{T}(u, size(Tile{T, Dims}))

"""
    buffer(a::NPUArray) -> Ptr{Cvoid}

The underlying XRT buffer-object handle, as passed to the shim's launch. See
[`run!`](@ref).
"""
buffer(a::NPUArray) = a.bo

# --- AbstractArray interface -------------------------------------------------
# `eltype`, `ndims` and `length` come from `AbstractGPUArray{T,N} <: AbstractArray`.

Base.size(a::NPUArray) = size(a.data)
Base.IndexStyle(::Type{<:NPUArray}) = IndexLinear()

# Scalar access keeps the host mapping coherent with the device: a read syncs the
# buffer back first, a write syncs it out after.
function Base.getindex(a::NPUArray, i::Int)
    @boundscheck checkbounds(a, i)
    GPUArraysCore.assertscalar("getindex")
    _xrt_bo_sync_from_device(a.bo)
    return @inbounds a.data[i]
end

function Base.setindex!(a::NPUArray{T}, v, i::Int) where {T}
    @boundscheck checkbounds(a, i)
    GPUArraysCore.assertscalar("setindex!")
    @inbounds a.data[i] = convert(T, v)
    _xrt_bo_sync_to_device(a.bo)
    return a
end

Base.similar(a::NPUArray{T}) where {T} = NPUArray{T}(undef, size(a))
Base.similar(a::NPUArray, ::Type{S}) where {S} = NPUArray{S}(undef, size(a))
Base.similar(a::NPUArray, ::Type{S}, dims::Dims) where {S} = NPUArray{S}(undef, dims)
Base.similar(::Type{<:NPUArray{T}}, dims::Dims) where {T} = NPUArray{T}(undef, dims)

# --- host <-> device transfer ------------------------------------------------

"""
    Array(a::NPUArray) -> Array

Copy an NPU-resident buffer back to a host `Array`.
"""
function Base.Array(a::NPUArray)
    _xrt_bo_sync_from_device(a.bo)
    return copy(a.data)
end
Base.collect(a::NPUArray) = Array(a)

Base.copyto!(dst::AbstractArray, src::NPUArray) = copyto!(dst, Array(src))

# Displaying follows the GPUArrays convention: copy to the host once and let Base
# render that, rather than scalar-indexing the device buffer element by element.
Base.print_array(io::IO, a::NPUArray) = Base.print_array(io, Array(a))

# --- Adapt: launch-argument conversion (NPUArray -> Tile) --------------------
# The IRON analogue of `CUDACore.KernelAdaptor`/`cuTile.KernelAdaptor`: adapting a
# host `NPUArray` yields the kernel-side `Tile` type the compiler infers against.

"""
    KernelAdaptor

`Adapt.jl` adaptor that converts a host launch argument to its kernel-side form: an
`NPUArray{T,N}` becomes the `Tile{T,Tuple{dims...}}` a kernel is compiled against.
"""
struct KernelAdaptor end

Adapt.adapt_storage(::KernelAdaptor, a::NPUArray{T, N}) where {T, N} =
    Tile{T, Tuple{size(a)...}}

"""
    kernelconvert(x)

Convert a launch argument to its kernel-side form via `Adapt.adapt` with
[`KernelAdaptor`](@ref). Mirrors `CUDACore.cudaconvert`.
"""
kernelconvert(x) = adapt(KernelAdaptor(), x)

# Derive a design's argument tiles straight from the host buffers, so a design can
# be described with the arrays it will run on rather than by spelling out each
# `Tile` type. Complements the explicit `Vector{Type}` form in `dataflow.jl`.
Program(device::AIEDevice, rt::Runtime, args::AbstractVector{<:NPUArray}; name::AbstractString = "main") =
    Program(device, rt, Type[kernelconvert(a) for a in args]; name)
