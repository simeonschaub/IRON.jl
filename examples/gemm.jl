# A tiled single-core GEMM, C = A * B, with the micro-kernels written in Julia and the
# schedule expressed with `@iron`'s `for` form.
#
# This is the minimal tiled *reduction*: (m, k) x (k, n) tiles stream through object
# FIFOs and reduce on one core into an (m, n) output tile. The `@iron for` loop nest
# spells out the schedule -- the outer loop is the output-tile iteration and the nested
# `@reduce for` is the accumulation -- and IRON wires up the FIFOs, worker and host DMA:
#
#     for each (mi, nj) output tile:      # the space loop
#         gemm_zero!(C)                    # @init
#         for each kk reduction tile:      # the @reduce loop
#             gemm_acc!(A, B, C)           # C += A * B
#
# A tile is column-major (Julia's convention), so the vectors run down columns: the
# accumulator carries a column of C, `gemm_acc!` loads a column of A and broadcasts a
# scalar of B, and the whole thing needs no transpose and no operand swap.
#
#   IRON_RUN=1 julia --project examples/gemm.jl

using IRON
using BFloat16s: BFloat16

# The row tile `m` is the vector width -- 16 f32/i32 lanes on AIE2's 512-bit
# registers -- so one `vload` fills a hardware vector. Larger tiles want the
# micro-kernel to sub-tile `m` the way mm.cc does with its `r` dimension.
const M, K, N = 64, 64, 64
const m, k, n = 16, 32, 16

"""
    gemm_zero!(c)

Clear an output tile, a column at a time. Run once per output tile by `@init`.
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
    # Small integers, exact in bf16, so the product can be compared for equality.
    a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
    b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{Tacc}(undef, Tile{Tacc, Tuple{M, N}})

    # C = A * B, in (m, k) x (k, n) tiles reduced on one core. Tile shapes are inferred
    # from each buffer and the extents of the axes indexing it (e.g. da is M x K indexed
    # by (mi, kk) of extent (M/m, K/k), giving an m x k tile).
    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, m), nj in 1:div(N, n)
        @init gemm_zero!(dc)
        @reduce for kk in 1:div(K, k)
            gemm_acc!(In(da)[mi, kk], In(db)[kk, nj], Out(dc)[mi, nj])
        end
    end

    result = Array(dc)
    expected = Tacc.(Float32.(a) * Float32.(b))
    if result == expected
        println("NPU gemm ($(M)x$(K)x$(N), tiles $(m)x$(k)x$(n)) matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
    end
else
    println("Tiled single-core GEMM: C = A * B, $(M)x$(K)x$(N) in $(m)x$(k)x$(n) tiles.")
    println("Run on an NPU with:  IRON_RUN=1 julia --project examples/gemm.jl")
end
