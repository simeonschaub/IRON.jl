# The host-side array: an NPU-resident XRT buffer wearing the JuliaGPU
# `AbstractGPUArray` interface, so IRON composes with the rest of the ecosystem
# the way CUDA.jl, oneAPI.jl and Metal.jl do -- `Adapt` for converting a launch
# argument to its kernel-side view, and GPUArraysCore's scalar-indexing guard.
#
# It deliberately depends on GPUArraysCore (the marker type + scalar guard) rather
# than GPUArrays.jl: the NPU runs fixed compiled `@iron` designs, not arbitrary
# KernelAbstractions kernels, so GPUArrays' generic broadcast/mapreduce have no
# backend to lower onto. `NPUArray` is thus a device *buffer handle* -- allocation,
# host<->device copies, and the launch-argument adaptor -- not a general compute
# array; the show methods below stand in for the ones GPUArrays.jl would provide.
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
    bo::Union{XRT.XRTWrap.BO, Nothing}  # XRT buffer object (owned; freed with the array by the GC)
    data::Array{T, N}   # host mapping of the buffer's memory (own = false)
end

function NPUArray(A::AbstractArray{T, N}) where {T, N}
    a = _npu_empty(T, size(A))
    copyto!(a.data, A)
    _bo_sync_to_device(a.bo)
    return a
end

NPUArray{T}(u::UndefInitializer, dims::Integer...) where {T} = NPUArray{T}(u, map(Int, dims))

# Genuinely uninitialized: no zero-fill, no sync. Outputs are overwritten by the launch,
# and an input is filled by the caller (`NPUArray(A)`, indexing, or `copyto!`).
NPUArray{T}(::UndefInitializer, dims::Dims{N}) where {T, N} = _npu_empty(T, dims)

# Allocate a buffer shaped like a kernel tile, the common case for a design output.
NPUArray{T}(u::UndefInitializer, ::Type{Tile{T, Dims}}) where {T, Dims} =
    NPUArray{T}(u, size(Tile{T, Dims}))

"""
    buffer(a::NPUArray) -> XRT.XRTWrap.BO

The underlying XRT buffer object, as passed to the launch. See
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
    _bo_sync_from_device(a.bo)
    return @inbounds a.data[i]
end

function Base.setindex!(a::NPUArray{T}, v, i::Int) where {T}
    @boundscheck checkbounds(a, i)
    GPUArraysCore.assertscalar("setindex!")
    @inbounds a.data[i] = convert(T, v)
    _bo_sync_to_device(a.bo)
    return a
end

# Base derives the 1- and 2-arg `similar` from these.
Base.similar(a::NPUArray, ::Type{S}, dims::Dims) where {S} = NPUArray{S}(undef, dims)
Base.similar(::Type{<:NPUArray{T}}, dims::Dims) where {T} = NPUArray{T}(undef, dims)

# --- host <-> device transfer ------------------------------------------------

"""
    Array(a::NPUArray) -> Array

Copy an NPU-resident buffer back to a host `Array`.
"""
function Base.Array(a::NPUArray)
    _bo_sync_from_device(a.bo)
    return copy(a.data)
end
Base.collect(a::NPUArray) = Array(a)

# Device <-> host and device <-> device copies. Each goes through the host mappings and
# syncs the affected buffers, rather than Base's element-wise fallback -- which would copy
# one element at a time through `getindex`/`setindex!` and trip the scalar-indexing guard.
Base.copyto!(dst::AbstractArray, src::NPUArray) = copyto!(dst, Array(src))

function Base.copyto!(dst::NPUArray, src::AbstractArray)
    copyto!(dst.data, src)
    _bo_sync_to_device(dst.bo)
    return dst
end

function Base.copyto!(dst::NPUArray, src::NPUArray)
    _bo_sync_from_device(src.bo)
    copyto!(dst.data, src.data)
    _bo_sync_to_device(dst.bo)
    return dst
end

# A device-resident copy (the buffer is duplicated on the NPU, not brought to the host).
Base.copy(a::NPUArray) = copyto!(_npu_empty(eltype(a), size(a)), a)

# Show via one host copy (the GPUArrays convention), not per-element getindex, which
# would trip the scalar-indexing guard. These are the entry points Base's show uses.
Base.print_array(io::IO, a::NPUArray) = Base.print_array(io, Array(a))
Base._show_nonempty(io::IO, a::NPUArray, prefix::String) = Base._show_nonempty(io, Array(a), prefix)
Base._show_empty(io::IO, a::NPUArray) = Base._show_empty(io, Array(a))
Base.show_vector(io::IO, a::NPUArray, args...) = Base.show_vector(io, Array(a), args...)

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
