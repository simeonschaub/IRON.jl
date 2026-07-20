# A GEMM whose micro-kernel is the hardware matmul (`vmatmul`) over a *sub-tiled* operand:
# `dims_to_stream` (via `L2(...; blocks=(r,s))`) streams each input tile block-columnar, so
# the core loads several 4x8 / 8x4 matmul blocks contiguously out of one bigger L1 tile and
# does several `vmatmul`s per streamed tile -- amortizing the DMA that made gemm_mm.jl
# overhead-bound. The output tile is kept a single 4x4 block (m=n=4), so C needs no layout
# transform; sub-tiling the output (m>4) is the next step.
#
#   IRON_RUN=1 julia --project examples/gemm_mm_tiled.jl

using IRON
using BFloat16s: BFloat16

# Operand tiles: A is m x k = 4 x 16 (one 4-row block, two 8-col k-blocks), B is k x n =
# 16 x 4 (two 8-row k-blocks, one 4-col block), C is 4 x 4 (one block).
const m, k, n = 4, 16, 4
const KB = k ÷ 8          # matmul k-blocks per tile (k / s)
const M, K, N = 64, 32, 64

function gemm_mm_zero!(c::Tile{Float32, Tuple{4, 4}})
    z = zero(Vec{4, Float32})
    for j in 1:4
        vstore!(z, c, 1, j)
    end
    return nothing
end

# `c += a * b`, reducing over the tile's k-blocks. `a`/`b` arrive block-columnar (each 4x8 /
# 8x4 block a column of the `(32, KB)` L1 tile), so block `kb` is `vload(Mat, tile, 1, kb)`.
function gemm_mm_tiled!(
        a::Tile{BFloat16, Tuple{32, KB}}, b::Tile{BFloat16, Tuple{32, KB}},
        c::Tile{Float32, Tuple{4, 4}},
    )
    # Unrolled over the KB=2 matmul k-blocks: a loop-carried vector accumulator becomes a
    # vector PHI that trips Peano's AIE2P pre-legalizer combiner, so keep it straight-line.
    acc = vload(Mat{4, 4, Float32}, c, 1, 1)
    acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, 1), vload(Mat{8, 4, BFloat16}, b, 1, 1), acc)
    acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, 2), vload(Mat{8, 4, BFloat16}, b, 1, 2), acc)
    vstore!(acc, c, 1, 1)
    return nothing
end

const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
    b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{Float32}(undef, Tile{Float32, Tuple{M, N}})

    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, m), nj in 1:div(N, n)
        @cores nj
        @init gemm_mm_zero!(dc)
        @reduce for kk in 1:div(K, k)
            gemm_mm_tiled!(
                L2(In(da); blocks = (4, 8))[mi, kk],
                L2(In(db); blocks = (8, 4))[kk, nj],
                L2(Out(dc))[mi, nj],
            )
        end
    end

    result = Array(dc)
    expected = Float32.(a) * Float32.(b)
    if result == expected
        println("NPU vmatmul (sub-tiled) gemm ($(M)x$(K)x$(N), $(div(N, n)) cores) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println("Sub-tiled vmatmul GEMM (dims_to_stream block-columnar inputs). Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/gemm_mm_tiled.jl")
end
