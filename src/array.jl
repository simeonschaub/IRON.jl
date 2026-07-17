# The host-side array: an NPU-resident XRT buffer wearing the JuliaGPU
# `AbstractGPUArray` interface, so IRON composes with the rest of the ecosystem the
# way CUDA.jl, oneAPI.jl and Metal.jl do -- `Adapt` for converting a launch
# argument to its kernel-side view, and GPUArraysCore's scalar-indexing guard.
#
# This is the host counterpart to `Tile`, the *device*-side view a kernel sees.
# A host `NPUArray{T,N}` adapts to the kernel `Tile{T,Tuple{dims...}}`, exactly as
# a `CuArray` adapts to a `CuDeviceArray`. The buffer itself stays on the Python
# side (an XRT tensor); `NPUArray` is the typed Julia handle to it.

"""
    NPUArray{T,N} <: AbstractGPUArray{T,N}

An `N`-dimensional array of `T` living in NPU-resident memory, backed by an XRT
buffer.

Construct one by copying a host array with `NPUArray(A)`, or allocate a zeroed
buffer of a given shape with `NPUArray{T}(undef, dims...)` (or, shaped like a
kernel tile, `NPUArray{T}(undef, Tile{T,Dims})`). Copy the contents back to the
host with `Array(x)`.

Pass `NPUArray`s to [`run!`](@ref) as the design's buffers. As a launch argument
each one adapts to the kernel's [`Tile`](@ref) view via `Adapt.adapt`.

Scalar indexing is disallowed by default -- touching one element still crosses to
the host and back through the Python/XRT stack -- so wrap any deliberate scalar
access in `@allowscalar` (re-exported from GPUArraysCore).
"""
struct NPUArray{T, N} <: AbstractGPUArray{T, N}
    buf::Py
    dims::NTuple{N, Int}
end

"""
    buffer(a::NPUArray) -> Py

The underlying XRT tensor. This is what the Python JIT design is called with; see
[`run!`](@ref).
"""
buffer(a::NPUArray) = a.buf

# Copy a host array into a fresh NPU buffer. `dtype` has to be passed even though
# the array already carries it: the tensor constructor defaults to uint32 and sizes
# the XRT buffer from the kwarg rather than from the data, then fails copying into
# the mistyped buffer.
function _device_tensor(A::AbstractArray{T}) where {T}
    dtype = numpy_dtype(T)
    host = np().array(host_values(A); dtype)
    return iron().tensor(host; dtype, device = "npu")
end

NPUArray(A::AbstractArray{T, N}) where {T, N} = NPUArray{T, N}(_device_tensor(A), size(A))

NPUArray{T}(::UndefInitializer, dims::Integer...) where {T} = NPUArray{T}(undef, dims)

function NPUArray{T}(::UndefInitializer, dims::Dims{N}) where {T, N}
    buf = iron().zeros(dims...; dtype = numpy_dtype(T), device = "npu")
    return NPUArray{T, N}(buf, dims)
end

# Allocate a buffer shaped like a kernel tile, the common case for a design output.
NPUArray{T}(u::UndefInitializer, ::Type{Tile{T, Dims}}) where {T, Dims} =
    NPUArray{T}(u, size(Tile{T, Dims}))

# --- AbstractArray interface -------------------------------------------------
# `eltype`, `ndims` and `length` come from `AbstractGPUArray{T,N} <: AbstractArray`.

Base.size(a::NPUArray) = a.dims
Base.size(a::NPUArray, d::Integer) = d <= ndims(a) ? a.dims[d] : 1

Base.IndexStyle(::Type{<:NPUArray}) = IndexCartesian()

# Scalar access is forwarded straight to the XRT tensor: PythonCall turns
# `getindex`/`setindex!` on a `Py` into the tensor's `__getitem__`/`__setitem__`,
# which handle device<->host synchronisation themselves (`__setitem__` syncs back
# to the device after the write -- going through `.numpy()` would drop that sync).
# Indices are shifted to the tensor's 0-based convention; the logical axis order
# matches `Array(a)`.
function Base.getindex(a::NPUArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(a, I...)
    GPUArraysCore.assertscalar("getindex")
    return pyconvert(T, a.buf[map(i -> i - 1, I)...])
end

function Base.setindex!(a::NPUArray{T, N}, v, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(a, I...)
    GPUArraysCore.assertscalar("setindex!")
    a.buf[map(i -> i - 1, I)...] = convert(T, v)
    return a
end

Base.similar(a::NPUArray{T}) where {T} = NPUArray{T}(undef, a.dims)
Base.similar(a::NPUArray, ::Type{S}) where {S} = NPUArray{S}(undef, a.dims)
Base.similar(a::NPUArray, ::Type{S}, dims::Dims) where {S} = NPUArray{S}(undef, dims)
Base.similar(::Type{<:NPUArray{T}}, dims::Dims) where {T} = NPUArray{T}(undef, dims)

# --- host <-> device transfer ------------------------------------------------

"""
    Array(a::NPUArray) -> Array

Copy an NPU-resident buffer back to a host `Array`.
"""
Base.Array(a::NPUArray) = pyconvert(Array, a.buf.numpy())
Base.collect(a::NPUArray) = Array(a)

Base.copyto!(dst::AbstractArray, src::NPUArray) = copyto!(dst, Array(src))

# Contents are deliberately not printed: displaying them would scalar-index the
# whole buffer back from the device.
Base.show(io::IO, a::NPUArray{T, N}) where {T, N} =
    print(io, "NPUArray{", T, ", ", N, "}(", size(a), ")")

function Base.show(io::IO, ::MIME"text/plain", a::NPUArray{T}) where {T}
    print(io, join(size(a), "×"), " NPUArray{", T, "} (device-resident; ")
    print(io, "Array(x) to copy to host)")
    return nothing
end

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
    Tile{T, Tuple{a.dims...}}

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
