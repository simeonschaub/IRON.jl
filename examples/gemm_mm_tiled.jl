# A GEMM whose micro-kernel is the hardware matmul (`vmatmul`), sub-tiled in m, k and n with a
# BIG per-core output tile (128 x 16). Growing m (not n) keeps the core count at N/n -- so the
# full array stays busy -- while making each core process far fewer, larger tiles: that
# amortizes the per-tile DMA/descriptor overhead that bounds the small-tile design, pushing the
# core into the compute-bound regime where `vmatmul`'s 128-MAC/op density actually pays off.
# Only possible because the k accumulation is now a LOOP (partial sum in the C tile, no PHI).
#
#   IRON_RUN=1 julia --project examples/gemm_mm_tiled.jl
#
# Tiles: A = 128 x 64 (32 m-blocks x 8 k-blocks of 4x8), B = 64 x 16 (8 k-blocks x 4 n-blocks of
# 8x4), C = 128 x 16 (32 m x 4 n blocks of 4x4). Block index orders: A (mb outer, kb inner) =
# mb*8+kb; B (kb outer, nb inner) = kb*4+nb; C (mb outer, nb inner) = mb*4+nb.

using IRON
using BFloat16s: BFloat16

const M, K, N = 256, 128, 32

# Clear the block-columnar C core tile: 128 columns of 16 = the 128 4x4 output blocks.
function gemm_mm_zero!(c::Tile{Float32, Tuple{16, 128}})
    z = zero(Vec{16, Float32})
    for j in 1:128
        vstore!(z, c, 1, j)
    end
    return nothing
end

# `c += a * b`: LOOP the 128 output blocks (32 m x 4 n), each an independent 4x4 C location, but
# UNROLL the 8 k-blocks straight-line -- the partial `acc` is held in a register across the k
# chain (loaded once, stored once per output block, not per k-block), so C L1 traffic drops 8x.
# Straight-line SSA has no loop-carried vector value, so no PHI for Peano's AIE2P combiner (the
# crash is specific to an scf.for vector iter-arg, which we still avoid -- the output loop's
# `acc` never crosses an iteration). Block indices: A mb*8+kb, B kb*4+nb, C mb*4+nb.
function gemm_mm_tiled!(
        a::Tile{BFloat16, Tuple{32, 256}}, b::Tile{BFloat16, Tuple{32, 32}},
        c::Tile{Float32, Tuple{16, 128}},
    )
    for mb in 0:31, nb in 0:3
        bi = mb * 4 + nb
        acc = vload(Mat{4, 4, Float32}, c, 1, bi + 1)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + 1), vload(Mat{8, 4, BFloat16}, b, 1, nb + 1), acc)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + 2), vload(Mat{8, 4, BFloat16}, b, 1, 4 + nb + 1), acc)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + 3), vload(Mat{8, 4, BFloat16}, b, 1, 8 + nb + 1), acc)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + 4), vload(Mat{8, 4, BFloat16}, b, 1, 12 + nb + 1), acc)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + 5), vload(Mat{8, 4, BFloat16}, b, 1, 16 + nb + 1), acc)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + 6), vload(Mat{8, 4, BFloat16}, b, 1, 20 + nb + 1), acc)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + 7), vload(Mat{8, 4, BFloat16}, b, 1, 24 + nb + 1), acc)
        acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + 8), vload(Mat{8, 4, BFloat16}, b, 1, 28 + nb + 1), acc)
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

    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, 128), nj in 1:div(N, 16)
        @cores nj
        @init gemm_mm_zero!(dc)
        @reduce for kk in 1:div(K, 64)
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
        println("NPU vmatmul (big 128x16 tile) gemm ($(M)x$(K)x$(N), $(div(N, 16)) cores) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println("Big-tile (64x16) sub-tiled vmatmul GEMM. Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/gemm_mm_tiled.jl")
end
