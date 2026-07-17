# A vectorized matrix multiply, reaching the AIE vector unit from Julia.
#
# The scalar unit on an AIE2 core cannot multiply floats: examples/diagnose.jl
# shows an f32 matmul returning wrong data while the same kernel over integers is
# correct. Float throughput lives in the vector unit, which the C++ kernels reach
# through `aie::mmul`. This gets there from Julia without emitting a single
# `aievec` op, because aiecc already runs `convert-vector-to-aievec` over every
# AIE2/AIE2p core, and that pipeline "ingests arbitrary MLIR Vector code":
#
#     vector.fma  ->  aievec.mac_elem  ->  the MAC intrinsic
#
# So the job is to emit the `vector` dialect, and `SIMD.Vec{N,T}` is how a kernel
# says that. Its arithmetic would normally inline to `llvmcall` carrying a literal
# LLVM IR string, which means nothing here; IRON's overlay method table redirects
# the operators to intrinsics first (src/simd.jl).
#
# Run with the MLIR-AIE ironenv python so the compile/run half can find `aie`:
#   JULIA_PYTHONCALL_EXE=/path/to/mlir-aie/ironenv/bin/python julia --project examples/matmul_vectorized.jl

using IRON
using SIMD: Vec

# The vector width is the hardware's, not the matrix's. convert-vector-to-aievec
# lowers vector.fma only for f32 at 16 lanes, and bf16 at 16 or 32 -- AIE2's vector
# registers are 512 bits, so 16 f32. A vector<8xf32> matches no pattern and aiecc
# stops with "failed to legalize operation 'vector.fma'". So the tile is as wide as
# one vector.
const M, K, N = 16, 16, 16

"""
    matmul_vec!(a, b, c)

`c = a * b`, a row of `c` at a time.

Each step broadcasts one element of `a` across the lanes, reads the matching row
of `b` as a vector, and multiply-accumulates: `N` outputs advance at once,
and the accumulator stays in a vector register across the whole `k` loop rather
than going near memory. This is the shape `aie::mmul` has, expressed in Julia.
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

function matmul_program(::Type{T}) where {T}
    A = Tile{T, Tuple{M, K}}
    of_a, of_b, of_c = ObjectFifo{A}("a"), ObjectFifo{A}("b"), ObjectFifo{A}("c")

    rt = Runtime()
    start!(rt, Worker(matmul_vec!, [consumer(of_a), consumer(of_b), producer(of_c)]))
    fill!(rt, producer(of_a), 1)
    fill!(rt, producer(of_b), 2)
    drain!(rt, consumer(of_c), 3)

    return Program(npu2, rt, [A, A, A])
end

# Bank-aware allocation, the default, silently overlaps the object FIFO buffers for
# a design with three FIFOs on one core; every matrix multiply under
# programming_examples/ passes this flag.
const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    T = Float32
    compiled = IRON.compile(matmul_program(T); aiecc_flags = AIECC_FLAGS)

    a = T[10i + j for i in 1:M, j in 1:K]
    b = T[i - 2j for i in 1:K, j in 1:N]
    da, db = IRON.device_array(a), IRON.device_array(b)
    dc = IRON.device_zeros(Tile{T, Tuple{M, N}})
    IRON.run!(compiled, da, db, dc)

    result = IRON.host_array(dc)
    expected = a * b
    if result == expected
        println("NPU vectorized matmul matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
        println("got:\n", result)
        println("expected:\n", expected)
    end
else
    println("the vector ops convert-vector-to-aievec consumes:\n")
    for T in (Float32, Int32)
        mlir = generate_mlir(matmul_program(T))
        println("  ", T, ":")
        for l in split(mlir, '\n')
            occursin(r"vector\.(load|store|broadcast|fma)", l) &&
                println("    ", strip(replace(l, r"^\s*%\w+ = " => "")))
        end
    end
end
