# The big-tile bf16 GEMM with the k accumulation carried in a REGISTER across the reduction
# loop (`acc` defined before the `kb` loop, reduced in it, stored once after) -- the natural
# form that previously crashed Peano's AIE2P PreLegalizer combiner on the loop-carried vector
# accumulator PHI. With that combiner bug fixed it should compile, and it avoids the per-k-block
# C-tile load/store the working kernel (gemm_mm_tiled.jl) uses -- ~8x less C L1 traffic.
#
#   IRON_RUN=1 julia --project examples/gemm_mm_tiled_phi.jl
#
# Same layout as gemm_mm_tiled.jl: A = 128x64, B = 64x16, C = 128x16, 8x8/4x8/8x4 -> 4x4 blocks,
# block indices A mb*8+kb, B kb*4+nb, C mb*4+nb.

using IRON
using BFloat16s: BFloat16

const M, K, N = 256, 128, 32

function gemm_mm_zero!(c::Tile{Float32, Tuple{16, 128}})
    z = zero(Vec{16, Float32})
    for j in 1:128
        vstore!(z, c, 1, j)
    end
    return nothing
end

# `c += a * b`: loop the 128 output blocks; for each, load the 4x4 partial ONCE, reduce the 8
# k-blocks with `acc` held in a register across the loop (loop-carried vector PHI), store ONCE.
function gemm_mm_tiled!(
        a::Tile{BFloat16, Tuple{32, 256}}, b::Tile{BFloat16, Tuple{32, 32}},
        c::Tile{Float32, Tuple{16, 128}},
    )
    for mb in 0:31, nb in 0:3
        bi = mb * 4 + nb
        acc = vload(Mat{4, 4, Float32}, c, 1, bi + 1)
        for kb in 0:7
            acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + kb + 1), vload(Mat{8, 4, BFloat16}, b, 1, kb * 4 + nb + 1), acc)
        end
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
        println("NPU vmatmul (register-carried k-loop) gemm ($(M)x$(K)x$(N), $(div(N, 16)) cores) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println("Register-carried-k big-tile vmatmul GEMM (was a Peano PHI crash). Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/gemm_mm_tiled_phi.jl")
end
