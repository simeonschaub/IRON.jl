# A tiled single-core GEMM, C = A * B, with the micro-kernel written in Julia.
#
# This is a port of single_core.py from mlir-aie's matrix_multiplication examples.
# The Python design streams (m, k) x (k, n) tiles through object FIFOs and reduces
# them on one core into an (m, n) output tile, binding an external C++ `mm.cc` for
# the multiply. Here the two micro-kernels -- `gemm_zero!` and `gemm_acc!` -- are
# ordinary Julia, and `gemm_program` (see src/gemm.jl) emits the same nested-acquire
# core loop plus the tiled host DMA:
#
#     for each (m, n) output tile:
#         acquire C; gemm_zero!(C)
#         for each of the K/k reduction tiles:
#             acquire A, B; gemm_acc!(A, B, C)   # C += A * B
#         release C
#
# A tile is column-major (Julia's convention), so the vectors run down columns: the
# accumulator carries a column of C, `gemm_acc!` loads a column of A and broadcasts a
# scalar of B, and the whole thing needs no transpose and no operand swap.
#
#   julia --project examples/gemm.jl

using IRON
using BFloat16s: BFloat16

# The row tile `m` is the vector width -- 16 f32/i32 lanes on AIE2's 512-bit
# registers -- so one `vload` fills a hardware vector. Larger tiles want the
# micro-kernel to sub-tile `m` the way mm.cc does with its `r` dimension.
const M, K, N = 64, 64, 64
const m, k, n = 16, 32, 16

"""
    gemm_zero!(c)

Clear an output tile, a column at a time.
"""
function gemm_zero!(c::Tile{Tacc, Tuple{m, n}}) where {Tacc, m, n}
    z = zero(Vec{m, Tacc})
    for j in 1:n
        vstore!(z, c, 1, j)
    end
    return nothing
end

"""
    gemm_acc!(a, b, c)

`c += a * b` for one tile, reading the running accumulator out of `c` and writing it
back, so a sequence of calls reduces over the k dimension. The same shape as
`matmul_vec!` in matmul_vectorized.jl, but accumulating into `c` rather than starting
from zero.
"""
function gemm_acc!(
        a::Tile{T, Tuple{m, k}}, b::Tile{T, Tuple{k, n}}, c::Tile{Tacc, Tuple{m, n}},
    ) where {T, Tacc, m, k, n}
    for j in 1:n
        acc = vload(Vec{m, Tacc}, c, 1, j)          # the running accumulator column
        for kk in 1:k
            av = vload(Vec{m, T}, a, 1, kk)          # a column of `a`
            bv = Vec{m, T}(b[kk, j])                 # a scalar of `b`, broadcast
            acc = muladd(Vec{m, Tacc}(av), Vec{m, Tacc}(bv), acc)
        end
        vstore!(acc, c, 1, j)
    end
    return nothing
end

const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    Tin, Tacc = BFloat16, Float32
    mlir = gemm_program(gemm_zero!, gemm_acc!, Tin, Tacc, M, K, N, m, k, n)
    compiled = IRON.compile(mlir, 3; flags = AIECC_FLAGS)  # A, B, C

    # Small integers, exact in bf16, so the product can be compared for equality.
    a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
    b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{Tacc}(undef, Tile{Tacc, Tuple{M, N}})
    IRON.run!(compiled, da, db, dc)

    result = Array(dc)
    expected = Tacc.(Float32.(a) * Float32.(b))
    if result == expected
        println("NPU gemm ($(M)x$(K)x$(N), tiles $(m)x$(k)x$(n)) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println(gemm_program(gemm_zero!, gemm_acc!, BFloat16, Float32, M, K, N, m, k, n))
end
