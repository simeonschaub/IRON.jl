# A vectorized matrix multiply, reaching the AIE vector unit from Julia.
#
# The scalar unit on an AIE2 core cannot multiply floats: examples/diagnose.jl
# shows an f32 matmul returning wrong data while the same kernel over integers is
# correct. Float throughput lives in the vector unit, which the C++ kernels reach
# through `aie::mmul`. This gets there without emitting a single `aievec` op,
# because aiecc already runs `convert-vector-to-aievec` over every AIE2/AIE2p core
# and that pipeline "ingests arbitrary MLIR Vector code":
#
#     arith.extf + vector.fma  ->  aievec.mac_elem  ->  the MAC intrinsic
#
# Two rules shape the kernel below, both from
# lib/Dialect/AIEVec/Transforms/VectorToAIEVecConversions.cpp:
#
#  1. The vector width is the hardware's, not the algorithm's. vector.fma lowers
#     only for f32 at 16 lanes, and bf16 at 16 or 32 -- AIE2's vector registers are
#     512 bits. A vector<8xf32> matches no pattern and aiecc stops with "failed to
#     legalize operation 'vector.fma'".
#
#  2. An f32 vector.fma lowers only when *both* operands come from an arith.extf on
#     bf16, and says so: "vector.fma operands are f32, and they don't come from
#     arith.extf on bf16; can't lower to aievec." There is no f32 multiplier
#     anywhere on the core -- the MAC multiplies bf16 and accumulates into f32,
#     which is the same fact as `_MM_COMBOS` listing (bfloat16, float32) and no f32
#     entry.
#
# So the operands are bf16, widened per vector, and the accumulator is f32: the
# mixed precision the hardware is built around. `Vec{16,BFloat16}` is how a kernel
# says it -- IRON's own Vec rather than SIMD.jl's, whose element type must come
# from a fixed list that BFloat16 is not on.
#
# Run with the MLIR-AIE ironenv python so the compile/run half can find `aie`:
#   JULIA_PYTHONCALL_EXE=/path/to/mlir-aie/ironenv/bin/python julia --project examples/matmul_vectorized.jl

using IRON
using BFloat16s: BFloat16

const M, K, N = 16, 16, 16   # one vector wide

"""
    matmul_bf16_vec!(a, b, c)

`c = a * b` with bf16 operands and an f32 accumulator, a row of `c` at a time.

Each step broadcasts one element of `a` across the lanes, reads the matching row
of `b` as a vector, widens both, and multiply-accumulates: `N` outputs advance at
once, and the accumulator stays in a vector register for the whole `k` loop rather
than going near memory. This is the shape `aie::mmul` has, written in Julia.
"""
function matmul_bf16_vec!(
        a::Tile{BFloat16, Tuple{M, K}}, b::Tile{BFloat16, Tuple{K, N}},
        c::Tile{Float32, Tuple{M, N}},
    ) where {M, K, N}
    for i in 1:M
        acc = zero(Vec{N, Float32})
        for k in 1:K
            av = Vec{N, BFloat16}(a[i, k])          # vector.broadcast
            bv = vload(Vec{N, BFloat16}, b, k, 1)   # vector.load
            # Vec{N,Float32}(::Vec{N,BFloat16}) is one arith.extf per vector, and
            # what makes the vector.fma below legal.
            acc = muladd(Vec{N, Float32}(av), Vec{N, Float32}(bv), acc)
        end
        vstore!(acc, c, i, 1)
    end
    return nothing
end

"""
    matmul_vec!(a, b, c)

`c = a * b` over one element type, for the integer case, which needs no widening:
`vector.fma` is floating-point only, so an integer multiply-accumulate lowers to
`arith.muli` + `arith.addi` over vectors, which have patterns of their own.
"""
function matmul_vec!(
        a::Tile{T, Tuple{M, K}}, b::Tile{T, Tuple{K, N}}, c::Tile{T, Tuple{M, N}}
    ) where {T, M, K, N}
    for i in 1:M
        acc = zero(Vec{N, T})
        for k in 1:K
            acc = muladd(Vec{N, T}(a[i, k]), vload(Vec{N, T}, b, k, 1), acc)
        end
        vstore!(acc, c, i, 1)
    end
    return nothing
end

function matmul_program(kernel, ::Type{Tin}, ::Type{Tacc}) where {Tin, Tacc}
    A = Tile{Tin, Tuple{M, K}}
    C = Tile{Tacc, Tuple{M, N}}
    of_a, of_b, of_c = ObjectFifo{A}("a"), ObjectFifo{A}("b"), ObjectFifo{C}("c")

    rt = Runtime()
    start!(rt, Worker(kernel, [consumer(of_a), consumer(of_b), producer(of_c)]))
    fill!(rt, producer(of_a), 1)
    fill!(rt, producer(of_b), 2)
    drain!(rt, consumer(of_c), 3)

    return Program(npu2, rt, [A, A, C])
end

# Bank-aware allocation, the default, silently overlaps the object FIFO buffers for
# a design with three FIFOs on one core; every matrix multiply under
# programming_examples/ passes this flag.
const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    program = matmul_program(matmul_bf16_vec!, BFloat16, Float32)
    compiled = IRON.compile(program; aiecc_flags = AIECC_FLAGS)

    # Small integers, exact in bf16, so the product can be compared for equality.
    a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
    b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = IRON.device_array(a), IRON.device_array(b)
    dc = IRON.device_zeros(Tile{Float32, Tuple{M, N}})
    IRON.run!(compiled, da, db, dc)

    result = IRON.host_array(dc)
    expected = Float32.(a) * Float32.(b)
    if result == expected
        println("NPU vectorized bf16 matmul matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
        println("got:\n", result)
        println("expected:\n", expected)
    end
else
    println("bf16 operands, f32 accumulator -- what convert-vector-to-aievec wants:\n")
    for l in split(generate_mlir(matmul_program(matmul_bf16_vec!, BFloat16, Float32)), '\n')
        occursin(r"vector\.(load|store|broadcast|fma)|arith\.extf", l) &&
            println("  ", strip(replace(l, r"^\s*%\w+ = " => "")))
    end

    println("\nintegers need no widening:\n")
    for l in split(generate_mlir(matmul_program(matmul_vec!, Int32, Int32)), '\n')
        occursin(r"vector\.(load|store|broadcast)|arith\.(muli|addi) .*vector", l) &&
            println("  ", strip(replace(l, r"^\s*%\w+ = " => "")))
    end
end
