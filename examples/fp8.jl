# FP8 tiles: 8-bit floats as a storage format.
#
# FP8 is not a type you compute in -- there is no FP8 arithmetic, on this hardware
# or on a CPU. It is a way to keep weights and activations small in memory, and
# every value is widened to something real before it is used. This example shows
# the two halves of that: dequantizing a tile of FP8 into f32, and a matmul that
# widens each operand on load.
#
# The interesting part is that `Float32(a[i])` is one instruction. The Julia FP8
# packages implement conversion in software, by unpacking the bits and
# renormalizing, because a CPU has no FP8 either; inferring a kernel against those
# methods buries a single hardware conversion under a few hundred integer ops. IRON
# infers kernels under its own AbstractInterpreter with an overlay method table
# (src/interpreter.jl) that replaces just those conversions, so what reaches MLIR
# is `arith.extf f8E4M3FN to f32`.
#
# What runs on hardware today: the dequantize kernel is conversion and stores only.
# The matmul is printed rather than run, because its accumulate is scalar f32
# arithmetic, which the core does not execute correctly -- see examples/matmul.jl.
#
# Run with the MLIR-AIE ironenv python so the compile/run half can find `aie`:
#   JULIA_PYTHONCALL_EXE=/path/to/mlir-aie/ironenv/bin/python julia --project examples/fp8.jl

using IRON
using DLFP8Types: Float8_E4M3FN, Float8_E5M2

const N = 1024
const M = 8

"""
    dequantize!(a, c)

Widen a tile of FP8 into f32, one `arith.extf` per element.

The reverse -- `c[i] = Float8_E4M3FN(a[i])` on an f32 tile -- is one `arith.truncf`
and quantizes instead.
"""
function dequantize!(a::Tile{F, Tuple{N}}, c::Tile{Float32, Tuple{N}}) where {F, N}
    for i in 1:N
        c[i] = Float32(a[i])
    end
    return nothing
end

"""
    matmul_fp8!(a, b, c)

`c = a * b` with FP8 operands widened into an f32 accumulator.
"""
function matmul_fp8!(
        a::Tile{F, Tuple{M, K}}, b::Tile{F, Tuple{K, N}}, c::Tile{Float32, Tuple{M, N}}
    ) where {F, M, K, N}
    for i in 1:M, j in 1:N
        acc = zero(Float32)
        for k in 1:K
            acc += Float32(a[i, k]) * Float32(b[k, j])
        end
        c[i, j] = acc
    end
    return nothing
end

function dequantize_program(::Type{F}) where {F}
    of_in = ObjectFifo{Tile{F, Tuple{N}}}("in")
    of_out = ObjectFifo{Tile{Float32, Tuple{N}}}("out")

    rt = Runtime()
    start!(rt, Worker(dequantize!, [consumer(of_in), producer(of_out)]))
    fill!(rt, producer(of_in), 1)
    drain!(rt, consumer(of_out), 2)

    return Program(npu2, rt, [Tile{F, Tuple{N}}, Tile{Float32, Tuple{N}}])
end

function matmul_program(::Type{F}) where {F}
    A = Tile{F, Tuple{M, M}}
    C = Tile{Float32, Tuple{M, M}}
    of_a, of_b, of_c = ObjectFifo{A}("a"), ObjectFifo{A}("b"), ObjectFifo{C}("c")

    rt = Runtime()
    start!(rt, Worker(matmul_fp8!, [consumer(of_a), consumer(of_b), producer(of_c)]))
    fill!(rt, producer(of_a), 1)
    fill!(rt, producer(of_b), 2)
    drain!(rt, consumer(of_c), 3)

    return Program(npu2, rt, [A, A, C])
end

const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    F = Float8_E4M3FN
    compiled = IRON.compile(dequantize_program(F); aiecc_flags = AIECC_FLAGS)

    # Values chosen to be exact in E4M3: small integers and negative powers of two.
    host = F[F((i % 8) - 4 + (i % 3) / 4) for i in 1:N]
    da = IRON.device_array(host)
    dc = IRON.device_zeros(Tile{Float32, Tuple{N}})
    IRON.run!(compiled, da, dc)

    result = IRON.host_array(dc)
    expected = Float32.(host)
    if result == expected
        println("NPU dequantize matches over $N elements")
    else
        println("MISMATCH in $(count(result .!= expected)) of $N")
        println("got:      ", result[1:8], " ...")
        println("expected: ", expected[1:8], " ...")
    end
else
    for F in (Float8_E4M3FN, Float8_E5M2)
        ir = generate_mlir(dequantize_program(F))
        conv = first(strip(l) for l in split(ir, '\n') if occursin("arith.extf", l))
        println(rpad(string(F), 16), "dequantize: ", replace(conv, r"^%\w+ = " => ""))
    end

    println("\nmatmul, FP8 operands into an f32 accumulator:")
    for l in split(generate_mlir(matmul_program(Float8_E4M3FN)), '\n')
        occursin(r"arith\.(extf|mulf|addf)|memref\.(load|store)", l) &&
            println("  ", strip(l))
    end
end
