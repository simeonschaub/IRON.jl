# A multi-core GEMM whose micro-kernel is the hardware matrix-multiply (`vmatmul`) rather
# than the scalar-broadcast `vfma` of gemm.jl. To keep the `Mat` loads whole-tile (a `Mat`
# load reads R*C *contiguous* elements, so it cannot slice a sub-block of a larger tile
# yet -- that needs `dims_to_stream`), each object-FIFO tile is exactly one matmul shape:
# 4x8 * 8x4 -> 4x4. That makes the tiles tiny, so this is a *correctness/throughput
# validation* of `vmatmul` in the streaming `@cores` + `L2` path, not the final design --
# bigger tiles (sub-tiled with `dims_to_stream`) are the next step.
#
#   IRON_RUN=1 julia --project examples/gemm_mm.jl

using IRON
using BFloat16s: BFloat16

# One hardware matmul tile.
const m, k, n = 4, 8, 4
const M, K, N = 64, 64, 64

"""Clear a 4x4 output tile, a column at a time (run once per output tile by `@init`)."""
function gemm_mm_zero!(c::Tile{Float32, Tuple{4, 4}})
    z = zero(Vec{4, Float32})
    for j in 1:4
        vstore!(z, c, 1, j)
    end
    return nothing
end

"""`c += a * b` for one 4x8 * 8x4 -> 4x4 tile, via the `vmatmul` MAC-array intrinsic,
reading the running accumulator out of `c` and back so a sequence reduces over k."""
function gemm_mm_acc!(
        a::Tile{BFloat16, Tuple{4, 8}}, b::Tile{BFloat16, Tuple{8, 4}},
        c::Tile{Float32, Tuple{4, 4}},
    )
    av = vload(Mat{4, 8, BFloat16}, a, 1, 1)
    bv = vload(Mat{8, 4, BFloat16}, b, 1, 1)
    acc = vload(Mat{4, 4, Float32}, c, 1, 1)
    vstore!(vmatmul(av, bv, acc), c, 1, 1)
    return nothing
end

const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
    b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{Float32}(undef, Tile{Float32, Tuple{M, N}})

    # N/n = 16 cores (4 memtile groups), every operand on L2.
    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, m), nj in 1:div(N, n)
        @cores nj
        @init gemm_mm_zero!(dc)
        @reduce for kk in 1:div(K, k)
            gemm_mm_acc!(L2(In(da))[mi, kk], L2(In(db))[kk, nj], L2(Out(dc))[mi, nj])
        end
    end

    result = Array(dc)
    expected = Float32.(a) * Float32.(b)
    if result == expected
        println("NPU vmatmul gemm ($(M)x$(K)x$(N), $(div(N, n)) cores) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println("Multi-core GEMM with the `vmatmul` matmul micro-kernel. Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/gemm_mm.jl")
end
