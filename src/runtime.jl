# Compiling and running a design on the NPU.
#
# Everything below the generated MLIR -- aiecc, the xclbin, XRT buffers and the
# kernel launch -- is MLIR-AIE's Python stack, driven here through PythonCall.
# Reimplementing it in Julia would buy nothing: the MLIR is the interesting part.
#
# The Python side must be the interpreter that has `aie` installed (the MLIR-AIE
# ironenv), so point PythonCall at it before loading IRON:
#
#     ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/mlir-aie/ironenv/bin/python"
#
# `iron.jit` accepts a path to a pre-written .mlir file, which is exactly the
# hand-off this package needs.

# Imported on first use rather than in `__init__`, so that generating MLIR works
# on a machine without the MLIR-AIE toolchain -- only compiling and running need it.
const PY_MODULES = Dict{String, Py}()

function pymodule(name::String)
    return get!(PY_MODULES, name) do
        try
            pyimport(name)
        catch e
            error(
                """
                IRON: could not import the Python module `$name`, which is needed to \
                compile and run designs (generating MLIR does not need it).

                Point PythonCall at the MLIR-AIE ironenv interpreter before loading IRON:
                    ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/mlir-aie/ironenv/bin/python"

                Original error: $e
                """
            )
        end
    end
end

iron() = pymodule("aie.iron")
np() = pymodule("numpy")

# Element types the NPU and numpy agree on.
const NUMPY_DTYPES = Dict{Type, String}(
    Int8 => "int8", Int16 => "int16", Int32 => "int32", Int64 => "int64",
    UInt8 => "uint8", UInt16 => "uint16", UInt32 => "uint32", UInt64 => "uint64",
    Float16 => "float16", Float32 => "float32", Float64 => "float64",
)

function numpy_dtype(::Type{T}) where {T}
    haskey(NUMPY_DTYPES, T) || error("IRON: no numpy dtype for $T")
    return getproperty(np(), Symbol(NUMPY_DTYPES[T]))
end

"""
    CompiledProgram

A [`Program`](@ref) lowered to MLIR and wrapped in an MLIR-AIE JIT design. Holds
the `.mlir` on disk because that file is what the Python side compiles.
"""
struct CompiledProgram
    program::Program
    path::String
    design::Py
end

"""
    compile(program; path=nothing, kwargs...) -> CompiledProgram

Generate `program`'s MLIR, write it out, and hand it to MLIR-AIE's JIT. Keyword
arguments are forwarded to `aie.iron.jit` (`use_cache`, `aiecc_flags`, ...).

The xclbin is built lazily on the first [`run!`](@ref), and cached thereafter.
"""
function compile(p::Program; path::Union{Nothing, AbstractString} = nothing, kwargs...)
    mlir = generate_mlir(p)
    file = path === nothing ? tempname() * ".mlir" : String(path)
    write(file, mlir)
    design = iron().jit(pymodule("pathlib").Path(file); kwargs...)
    return CompiledProgram(p, file, design)
end

"""
    device_array(A) -> Py

Copy Julia array `A` into an NPU-resident XRT buffer.
"""
function device_array(A::AbstractArray{T}) where {T}
    # `iron.tensor` rejects a `dtype` kwarg that disagrees with a typed array, so
    # the dtype is fixed here, on the numpy side, and not passed again.
    host = np().array(collect(A); dtype = numpy_dtype(T))
    return iron().tensor(host; device = "npu")
end

"""
    device_zeros(::Type{Tile{T,Dims}}) -> Py

An NPU-resident XRT buffer shaped like the given tile, zero filled.
"""
function device_zeros(::Type{Tile{T, Dims}}) where {T, Dims}
    return iron().zeros(size(Tile{T, Dims})...; dtype = numpy_dtype(T), device = "npu")
end

"""
    run!(compiled, buffers...) -> nothing

Run the design on the NPU. `buffers` are NPU-resident arrays -- one per runtime
sequence argument, in order -- as returned by [`device_array`](@ref) or
[`device_zeros`](@ref). Outputs are written in place.

The first call compiles the design; later calls reuse the cached xclbin.
"""
function run!(c::CompiledProgram, buffers::Py...)
    length(buffers) == length(c.program.argtypes) || error(
        "IRON: design takes $(length(c.program.argtypes)) buffers, got $(length(buffers))"
    )
    c.design(buffers...)
    return nothing
end

"""
    host_array(buffer) -> Array

Copy an NPU-resident buffer back to a Julia array.
"""
host_array(buffer::Py) = pyconvert(Array, buffer.numpy())
