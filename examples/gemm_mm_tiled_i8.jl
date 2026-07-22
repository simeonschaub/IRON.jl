# The int8 big-tile GEMM: the int8 counterpart of gemm_mm_tiled.jl. AIE2P's int8 matmul is
# 8x8 * 8x8 -> 8x8 (i32 acc, 512 MACs/op), so every matmul block is 8x8 and the accumulator
# is Int32. Same big per-core tile idea (grow m to 128, hold n=16 -> full N/16 core array,
# fewer/larger tiles) and the same PHI-free k-loop (partial in the C tile) as the bf16 kernel.
#
#   IRON_RUN=1 julia --project examples/gemm_mm_tiled_i8.jl
#
# Tiles: A = 128 x 64 (16 m-blocks x 8 k-blocks of 8x8), B = 64 x 16 (8 k-blocks x 2 n-blocks
# of 8x8), C = 128 x 16 (16 m x 2 n blocks of 8x8, i32). Block index orders: A (mb outer, kb
# inner) = mb*8+kb; B (kb outer, nb inner) = kb*2+nb; C (mb outer, nb inner) = mb*2+nb.

using IRON

const M, K, N = 256, 128, 32

# Clear the block-columnar C core tile: 32 columns of 64 i32 = the 32 8x8 output blocks.
function gemm_i8_zero!(c::Tile{Int32, Tuple{64, 32}})
    z = zero(Vec{64, Int32})
    for j in 1:32
        vstore!(z, c, 1, j)
    end
    return nothing
end

# `c += a * b`: loop the 32 output blocks (16 m x 2 n), loop the 8 k-blocks keeping the running
# 8x8 i32 partial in the C tile (load/accumulate/store per k-block) -- no loop-carried vector
# register, so no Peano AIE2P PHI crash. 256 vmatmuls/tile, each a 512-MAC int8 8x8x8.
function gemm_i8_acc!(
        a::Tile{Int8, Tuple{64, 128}}, b::Tile{Int8, Tuple{64, 16}},
        c::Tile{Int32, Tuple{64, 32}},
    )
    for mb in 0:15, nb in 0:1
        bi = mb * 2 + nb
        for kb in 0:7
            acc = vload(Mat{8, 8, Int32}, c, 1, bi + 1)
            acc = vmatmul(vload(Mat{8, 8, Int8}, a, 1, mb * 8 + kb + 1), vload(Mat{8, 8, Int8}, b, 1, kb * 2 + nb + 1), acc)
            vstore!(acc, c, 1, bi + 1)
        end
    end
    return nothing
end

const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    a = Int8[(i + j) % 7 for i in 1:M, j in 1:K]
    b = Int8[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{Int32}(undef, Tile{Int32, Tuple{M, N}})

    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, 128), nj in 1:div(N, 16)
        @cores nj
        @init gemm_i8_zero!(dc)
        @reduce for kk in 1:div(K, 64)
            gemm_i8_acc!(
                L2(In(da); blocks = (8, 8))[mi, kk],
                L2(In(db); blocks = (8, 8))[kk, nj],
                L2(Out(dc); blocks = (8, 8))[mi, nj],
            )
        end
    end

    result = Array(dc)
    expected = Int32.(a) * Int32.(b)
    if result == expected
        println("NPU int8 vmatmul (big 128x16 tile) gemm ($(M)x$(K)x$(N), $(div(N, 16)) cores) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println("Big-tile int8 sub-tiled vmatmul GEMM. Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/gemm_mm_tiled_i8.jl")
end
