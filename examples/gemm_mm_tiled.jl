# A GEMM whose micro-kernel is the hardware matmul (`vmatmul`) over tiles sub-tiled in m, k,
# AND n: `L2(...; blocks=(r,s))` streams each operand block-columnar via `dims_to_stream` (and
# un-blocks the output on the join). A bigger `(8, 16)` output tile means only N/16 cores and
# one C DMA covering many 4x4 output blocks -- the amortization that lets the MAC-array kernel
# pull ahead of the scalar-broadcast one.
#
#   IRON_RUN=1 julia --project examples/gemm_mm_tiled.jl
#
# Tiles: A = 8 x 16 (2 m-blocks x 2 k-blocks of 4x8), B = 16 x 16 (2 k-blocks x 4 n-blocks of
# 8x4), C = 8 x 16 (2 m x 4 n blocks of 4x4). Block index orders: A (mb outer, kb inner) =
# mb*2+kb; B (kb outer, nb inner) = kb*4+nb; C (mb outer, nb inner) = mb*4+nb.

using IRON
using BFloat16s: BFloat16

const M, K, N = 64, 32, 64

# Clear the block-columnar C core tile: 8 columns of 16 = the eight 4x4 output blocks.
function gemm_mm_zero!(c::Tile{Float32, Tuple{16, 8}})
    z = zero(Vec{16, Float32})
    for j in 1:8
        vstore!(z, c, 1, j)
    end
    return nothing
end

# `c += a * b`. Loop over the 8 output blocks (acc is loaded/stored each iteration, never
# loop-carried, so no vector PHI); the two k-blocks are unrolled.
function gemm_mm_tiled!(
        a::Tile{BFloat16, Tuple{32, 4}}, b::Tile{BFloat16, Tuple{32, 8}},
        c::Tile{Float32, Tuple{16, 8}},
    )
    for mb in 0:1, nb in 0:3
        bi = mb * 4 + nb
        acc = vload(Mat{4, 4, Float32}, c, 1, bi + 1)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 2 + 1), vload(Mat{8, 4, BFloat16}, b, 1, nb + 1), acc)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 2 + 2), vload(Mat{8, 4, BFloat16}, b, 1, 4 + nb + 1), acc)
        vstore!(acc, c, 1, bi + 1)
    end
    return nothing
end

const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
    b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{Float32}(undef, Tile{Float32, Tuple{M, N}})

    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, 8), nj in 1:div(N, 16)
        @cores nj
        @init gemm_mm_zero!(dc)
        @reduce for kk in 1:div(K, 16)
            gemm_mm_tiled!(
                L2(In(da); blocks = (4, 8))[mi, kk],
                L2(In(db); blocks = (8, 4))[kk, nj],
                L2(Out(dc); blocks = (4, 4))[mi, nj],
            )
        end
    end

    result = Array(dc)
    expected = Float32.(a) * Float32.(b)
    if result == expected
        println("NPU vmatmul (m/k/n sub-tiled) gemm ($(M)x$(K)x$(N), $(div(N, 16)) cores) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println("m/k/n sub-tiled vmatmul GEMM. Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/gemm_mm_tiled.jl")
end
