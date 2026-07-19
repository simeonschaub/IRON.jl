# A feed-forward neural-network layer on the NPU, driven entirely by the `@iron`
# macro -- no hand-written object FIFOs, workers or host DMA.
#
#     Y = relu(X * W + b)
#
# `X` is a `(batch, in)` activation matrix, `W` an `(in, out)` weight matrix and `b`
# an `(out,)` bias. The layer is two NPU launches, one per `@iron` form:
#
#   1. the matmul `Z = X * W + b` as an `@iron for` tiled **reduction** -- the GEMM
#      shape, where an output tile is accumulated across the shared (`in`) dimension;
#   2. the activation `Y = relu(Z)` as an `@iron` **tiled map** -- one elementwise
#      kernel run over a grid of tiles, SIMT in spirit.
#
# The bias is folded into the matmul with the standard augmentation trick, so the
# reduction produces `X*W + b` directly and the activation stays a pure elementwise
# map (no awkward broadcast of a 1-D bias across a 2-D tile grid):
#
#     Xa = [X | 1]   (batch, in+1)      -- append a column of ones
#     Wa = [W ; bᵀ]  (in+1, out)        -- append the bias as a row
#     Xa * Wa = X*W + 1·bᵀ = X*W + b
#
# Everything is tiled to 16, the AIE2 vector width for f32/bf16-accumulate, so every
# column the kernels load is one 512-bit vector register. The operands are bf16 and
# the accumulator f32 -- the mixed precision the vector MAC is built around; see
# examples/matmul_vectorized.jl for why that (and not f32*f32) is what the hardware
# multiplies.
#
# Compiling and running need the AIE toolchain JLLs and an NPU, but no Python:
#   IRON_RUN=1 julia --project examples/feedforward_relu.jl
# Without IRON_RUN the CPU reference is computed and printed, so the file runs anywhere.

using IRON
using BFloat16s: BFloat16

const TILE = 16   # AIE2 vector width: 16 f32/i32 lanes per 512-bit register.

# --- kernels: ordinary Julia functions, inlined into the core by `@iron` ---------

"""
    zero_tile!(c)

Clear an accumulator tile, one column (one vector) at a time. Run once per output
tile by `@init`, before the reduction accumulates into it.
"""
function zero_tile!(c::Tile{T, Tuple{M, N}}) where {T, M, N}
    z = zero(Vec{M, T})
    for j in 1:N
        vstore!(z, c, 1, j)
    end
    return nothing
end

"""
    matmul_acc!(a, b, c)

`c += a * b`, accumulating into the held accumulator tile `c`. The step kernel of
the reduction: `c` carries the running sum across the `@reduce` (k) loop, so it is
loaded, updated and stored rather than overwritten.

Operands are bf16 and the accumulator f32. A tile is column-major, so the contiguous
vector is a *column*: `vload` reads down a column of `a`, `b[p, j]` is broadcast, and
each is widened bf16→f32 (the `arith.extf` the vector MAC requires) before the
multiply-accumulate.
"""
function matmul_acc!(
        a::Tile{T, Tuple{M, K}}, b::Tile{T, Tuple{K, N}}, c::Tile{Tacc, Tuple{M, N}}
    ) where {T, Tacc, M, K, N}
    for j in 1:N
        acc = vload(Vec{M, Tacc}, c, 1, j)      # running sum (zeroed by @init on the first step)
        for p in 1:K
            av = vload(Vec{M, T}, a, 1, p)      # a column of A, in bf16
            bv = Vec{M, T}(b[p, j])             # broadcast B[p, j], in bf16
            acc = muladd(Vec{M, Tacc}(av), Vec{M, Tacc}(bv), acc)  # widen, then MAC
        end
        vstore!(acc, c, 1, j)
    end
    return nothing
end

"""
    relu!(z, y)

`y = max(z, 0)` for f32 tiles, one column (one vector) at a time -- the activation,
as an elementwise map over tiles.

AIE's vector max (`aievec.max`) supports i8/i16/i32/bf16 but *not* f32, so the relu
uses the sign-bit trick rather than a float max: reinterpret each f32 vector's bits
as signed i32, where a negative float is a negative integer, so `max(bits, 0)` in i32
zeros exactly the negatives and leaves the non-negatives untouched. Reinterpreting
back gives an exact f32 relu that routes through the supported i32 max.
"""
function relu!(z::Tile{Float32, Tuple{M, N}}, y::Tile{Float32, Tuple{M, N}}) where {M, N}
    zeroi = zero(Vec{M, Int32})
    for j in 1:N
        bits = reinterpret(Int32, vload(Vec{M, Float32}, z, 1, j))
        vstore!(reinterpret(Float32, max(bits, zeroi)), y, 1, j)
    end
    return nothing
end

# --- problem data (plain Julia; small integers, exact in bf16 and f32) -----------

const BATCH, IN, OUT = 32, 31, 32   # in+1 = 32 is a multiple of TILE, giving 2 reduce steps

# Deterministic small weights; W has negatives so the relu actually clips.
X    = Float32[(i + j) % 5 for i in 1:BATCH, j in 1:IN]
W    = Float32[((i - j) % 4) - 1 for i in 1:IN, j in 1:OUT]
bias = Float32[(j % 3) - 1 for j in 1:OUT]

# The augmented bf16 operands that fold the bias into the matmul: Xa = [X | 1] and
# Wa = [W ; bᵀ]. Built with scalar comprehensions rather than a `BFloat16.(...)`
# broadcast -- broadcasting an f32→bf16 conversion fuses into an @simd loop that
# LLVM's x86 host backend cannot select at width 16 ("Cannot select: v16bf16 =
# insert_subvector"), so convert one element at a time, as matmul_vectorized.jl does.
Xa = BFloat16[j <= IN ? BFloat16(X[i, j]) : one(BFloat16) for i in 1:BATCH, j in 1:(IN + 1)]
Wa = BFloat16[i <= IN ? BFloat16(W[i, j]) : BFloat16(bias[j]) for i in 1:(IN + 1), j in 1:OUT]

# CPU reference: what the NPU layer should produce.
Yref = max.(X * W .+ bias', 0.0f0)

# Bank-aware allocation silently overlaps the FIFO buffers of a multi-FIFO core; the
# matmul examples all pass this flag (see the README / matmul_vectorized.jl).
const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    dXa = NPUArray(Xa)
    dWa = NPUArray(Wa)
    dZ  = NPUArray{Float32}(undef, Tile{Float32, Tuple{BATCH, OUT}})
    dY  = NPUArray{Float32}(undef, Tile{Float32, Tuple{BATCH, OUT}})

    # 1. Z = Xa * Wa  (= X*W + b), a tiled reduction over the shared (in+1) dimension.
    #    Output tiles are (bi, oj); the accumulator is held across the kk reduction.
    @iron stack_size = 3328 flags = AIECC_FLAGS for bi in 1:BATCH÷TILE, oj in 1:OUT÷TILE
        @init zero_tile!(dZ)
        matmul_acc!(In(dXa)[bi, kk], In(dWa)[kk, oj], Out(dZ)[bi, oj])
        @reduce kk = (IN + 1) ÷ TILE
    end

    # 2. Y = relu(Z), an elementwise map streamed over a grid of 16x16 tiles.
    @iron relu!(In(dZ)::Tile{Float32, Tuple{TILE, TILE}}, Out(dY)::Tile{Float32, Tuple{TILE, TILE}})

    Y = Array(dY)
    if Y == Yref
        println("feed-forward relu: PASS  (", BATCH, "x", IN, " · ", IN, "x", OUT, ")")
        println("  Y[1:4, 1:4] = ")
        show(stdout, "text/plain", Y[1:4, 1:4]); println()
    else
        println("feed-forward relu: MISMATCH in ", count(Y .!= Yref), " of ", length(Yref))
        println("  got:      ", Y[1:2, 1:6])
        println("  expected: ", Yref[1:2, 1:6])
    end
else
    println("feed-forward relu layer: Y = relu(X*W + b), ", BATCH, "x", IN, " · ", IN, "x", OUT)
    println("Run on an NPU with:  IRON_RUN=1 julia --project examples/feedforward_relu.jl")
    println()
    println("CPU reference Y[1:4, 1:6]:")
    show(stdout, "text/plain", Yref[1:4, 1:6]); println()
end
