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

"""
    ml_dtypes() -> Py

The `ml_dtypes` module, which supplies the element types numpy lacks natively --
bfloat16 and the FP8 formats. MLIR-AIE depends on it, so it is present wherever
`aie` is.
"""
ml_dtypes() = pymodule("ml_dtypes")

# Element types numpy has natively.
const NUMPY_DTYPES = Dict{Type, String}(
    Int8 => "int8", Int16 => "int16", Int32 => "int32", Int64 => "int64",
    UInt8 => "uint8", UInt16 => "uint16", UInt32 => "uint32", UInt64 => "uint64",
    Float16 => "float16", Float32 => "float32", Float64 => "float64",
)

"""
    numpy_dtype(T) -> Py

The numpy dtype for Julia element type `T`.

Overload this for an element type numpy does not have; `ml_dtypes` provides the
ones the NPU cares about, and the FP8 formats are attached that way in `ext/`.
"""
function numpy_dtype(::Type{T}) where {T}
    name = get(NUMPY_DTYPES, T, nothing)
    name === nothing && error(
        "IRON: no numpy dtype for $T; overload IRON.numpy_dtype to add one"
    )
    return getproperty(np(), Symbol(name))
end

numpy_dtype(::Type{Core.BFloat16}) = ml_dtypes().bfloat16

"""
    host_values(A) -> AbstractArray

`A` in a form PythonCall can hand to numpy.

An array whose element type numpy shares is passed through. The rest go via
`Float32`, which each of them converts to exactly, and numpy narrows back to the
target dtype on the way into the buffer -- PythonCall has no numpy counterpart for
a `BFloat16` or FP8 array to convert directly.
"""
host_values(A::AbstractArray) = collect(A)
host_values(A::AbstractArray{<:Union{Core.BFloat16}}) = Float32.(A)

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
    run!(compiled, arrays...) -> nothing

Run the design on the NPU. `arrays` are the NPU-resident [`NPUArray`](@ref)s -- one
per runtime sequence argument, in order. Outputs are written in place; copy them
back to the host with `Array`.

The first call compiles the design; later calls reuse the cached xclbin.
"""
function run!(c::CompiledProgram, arrays::NPUArray...)
    length(arrays) == length(c.program.argtypes) || error(
        "IRON: design takes $(length(c.program.argtypes)) buffers, got $(length(arrays))"
    )
    c.design(map(buffer, arrays)...)
    return nothing
end
