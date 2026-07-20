# A GEMM whose micro-kernel is the hardware matmul (`vmatmul`) over tiles sub-tiled in BOTH
# the reduction and the output: `L2(...; blocks=(r,s))` streams each operand block-columnar
# via `dims_to_stream` (and un-blocks the output on the join side), so one bigger tile feeds
# several `vmatmul`s and one C DMA covers several output blocks. This is the layout that lets
# the MAC-array kernel amortize DMA the way whole_array does.
#
#   IRON_RUN=1 julia --project examples/gemm_mm_tiled.jl
#
# Tiles: A = m x k = 8 x 16 (2 m-blocks x 2 k-blocks of 4x8), B = 16 x 4 (2 k-blocks of 8x4),
# C = 8 x 4 (2 m-blocks of 4x4, single n-block). The core loads block `b` of an operand as
# `vload(Mat, tile, 1, b)`; A blocks are (mb outer, kb inner), so block index = mb*2 + kb.

using IRON
using BFloat16s: BFloat16

const M, K, N = 64, 32, 64

# Clear the block-columnar C core tile (2 columns of 16 = the two 4x4 output blocks).
function gemm_mm_zero!(c::Tile{Float32, Tuple{16, 2}})
    z = zero(Vec{16, Float32})
    vstore!(z, c, 1, 1)
    vstore!(z, c, 1, 2)
    return nothing
end

# `c += a * b`, fully unrolled over the 2 output m-blocks x 2 k-blocks. A loop-carried vector
# accumulator would become a PHI that crashes Peano's AIE2P combiner, so keep it straight-line.
function gemm_mm_tiled!(
        a::Tile{BFloat16, Tuple{32, 4}}, b::Tile{BFloat16, Tuple{32, 2}},
        c::Tile{Float32, Tuple{16, 2}},
    )
    acc0 = vload(Mat{4, 4, Float32}, c, 1, 1)                                     # output block 0
    acc0 = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, 1), vload(Mat{8, 4, BFloat16}, b, 1, 1), acc0)
    acc0 = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, 2), vload(Mat{8, 4, BFloat16}, b, 1, 2), acc0)
    vstore!(acc0, c, 1, 1)

    acc1 = vload(Mat{4, 4, Float32}, c, 1, 2)                                     # output block 1
    acc1 = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, 3), vload(Mat{8, 4, BFloat16}, b, 1, 1), acc1)
    acc1 = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, 4), vload(Mat{8, 4, BFloat16}, b, 1, 2), acc1)
    vstore!(acc1, c, 1, 2)
    return nothing
end

const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
    b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{Float32}(undef, Tile{Float32, Tuple{M, N}})

    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, 8), nj in 1:div(N, 4)
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
        println("NPU vmatmul (m/k sub-tiled) gemm ($(M)x$(K)x$(N), $(div(N, 4)) cores) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println("Output-and-reduction sub-tiled vmatmul GEMM. Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/gemm_mm_tiled.jl")
end
