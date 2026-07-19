# A three-layer MLP classifier on the NPU, driven entirely by `@iron` -- two hidden
# relu layers and a softmax output, no hand-written FIFOs, workers or host DMA:
#
#     H1 = relu(X · W1 + B1)
#     H2 = relu(H1 · W2 + B2)
#     Y  = softmax(H2 · W3 + B3)      # per-sample class probabilities
#
# Each layer is two `@iron` launches, one per macro form:
#   * the linear part `A · W` is an `@iron for` tiled **reduction** (the GEMM shape);
#   * the activation is an `@iron` **tiled map** (relu, or softmax on the last layer).
#
# Layout is batch-first and column-major, so a tile's 16 rows are 16 batch samples on
# the vector lanes (SIMT in spirit) and a tile's columns are features. Softmax is then
# a per-sample reduction *over the class columns*, which needs no cross-lane shuffle --
# each lane computes its own sample's softmax independently.
#
# ## Getting to exp on the AIE
#
# Softmax is `exp` then normalise, and neither is a plain vector op here:
#
#   * **exp**: there is a real hardware exp, but only for bf16 -- `convert-vector-to-
#     aievec` lowers `math.exp` on a `Vec{16,BFloat16}` to the AIE exp intrinsic (an f32
#     `math.exp` lowers to nothing). So the exp is taken in bf16 and widened back to f32.
#   * **max** (for numerical stability): `aievec.max` has no f32 form either (i8/i16/i32/
#     bf16 only), so the per-sample max is taken in bf16.
#   * **reciprocal** (`1/sum`): `arith.divf` of a constant `1` over a vector lowers to
#     `aievec.inv`; a general `a/b` does not, so normalise as `e * (1/s)`.
#   * **multiply** (the normalise `e * 1/s`): a plain f32 vector multiply has no
#     lowering (`aievec.mul_elem` is bf16-only, like the MAC), so the factors go
#     through bf16 -- `mulf` over two `extf`-from-bf16 operands is `mul_elem(bf16)`.
#   * **sum**: an f32 vector add lowers to `aievec.add_elem` (which does support f32).
#
# So the softmax kernel below stays in f32 except where the hardware forces bf16 (the
# max, exp and the normalise multiply), which is also where bf16 precision is plenty.
#
# The linear layers keep the validated bf16-operand / f32-accumulate matmul from
# examples/feedforward_relu.jl; the hidden activations are emitted in bf16 so they feed
# the next matmul directly.
#
# Compiling and running need the AIE toolchain JLLs and an NPU, but no Python:
#   IRON_RUN=1 julia --project=examples examples/mlp_softmax.jl
# Without IRON_RUN the CPU reference is computed and printed, so the file runs anywhere.

using IRON
using BFloat16s: BFloat16
using Random

const TILE = 16   # AIE2 vector width: 16 f32/i32 lanes, 32 bf16 -- everything tiles to 16.

# --- kernels: ordinary Julia functions, inlined into the core by `@iron` -------------

"""
    zero_tile!(c)

Clear an accumulator tile, run once per output tile by `@init`.
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

`c += a * b`, accumulating into the held f32 accumulator `c` across the `@reduce`
loop. Operands are bf16, widened to f32 for the multiply-accumulate (the vector MAC's
mixed precision). A tile is column-major, so a column is the contiguous vector.
"""
function matmul_acc!(
        a::Tile{T, Tuple{M, K}}, b::Tile{T, Tuple{K, N}}, c::Tile{Tacc, Tuple{M, N}}
    ) where {T, Tacc, M, K, N}
    for j in 1:N
        acc = vload(Vec{M, Tacc}, c, 1, j)
        for p in 1:K
            av = vload(Vec{M, T}, a, 1, p)
            bv = Vec{M, T}(b[p, j])
            acc = muladd(Vec{M, Tacc}(av), Vec{M, Tacc}(bv), acc)
        end
        vstore!(acc, c, 1, j)
    end
    return nothing
end

"""
    relu_bias!(z, bias, y)

`y = relu(z + bias)`, narrowing the result to bf16 so it feeds the next matmul. The
bias is a co-tiled `(batch, features)` buffer (every row the same), which sidesteps
broadcasting a 1-D bias across a 2-D tile grid. The relu is a bf16 `max` (the type
`aievec.max` supports), exact because the output is bf16 anyway.
"""
function relu_bias!(
        z::Tile{Float32, Tuple{M, N}}, bias::Tile{Float32, Tuple{M, N}}, y::Tile{BFloat16, Tuple{M, N}}
    ) where {M, N}
    zerob = zero(Vec{M, BFloat16})
    for j in 1:N
        v = Vec{M, BFloat16}(vload(Vec{M, Float32}, z, 1, j) + vload(Vec{M, Float32}, bias, 1, j))
        vstore!(max(v, zerob), y, 1, j)
    end
    return nothing
end

"""
    softmax!(z, bias, y)

`y = softmax(z + bias)` per sample (per lane), reducing over the `N` class columns.
Three passes over the tile: the biased logits into `y`, then the per-sample max, the
exp and its sum, and finally the normalisation. See the file header for why the max
and exp are in bf16 while the sum and normalise are in f32.
"""
function softmax!(
        z::Tile{Float32, Tuple{M, N}}, bias::Tile{Float32, Tuple{M, N}}, y::Tile{Float32, Tuple{M, N}}
    ) where {M, N}
    # biased logits -> y (f32)
    for j in 1:N
        vstore!(vload(Vec{M, Float32}, z, 1, j) + vload(Vec{M, Float32}, bias, 1, j), y, 1, j)
    end
    # per-sample max over classes, in bf16 (aievec.max has no f32 form)
    m = Vec{M, BFloat16}(vload(Vec{M, Float32}, y, 1, 1))
    for j in 2:N
        m = max(m, Vec{M, BFloat16}(vload(Vec{M, Float32}, y, 1, j)))
    end
    # exp(logit - max) in bf16 (hardware exp), summed in f32
    s = zero(Vec{M, Float32})
    for j in 1:N
        e = Vec{M, Float32}(exp(Vec{M, BFloat16}(vload(Vec{M, Float32}, y, 1, j)) - m))
        vstore!(e, y, 1, j)
        s = s + e
    end
    # normalise: reciprocal (1/s -> aievec.inv), then multiply. A plain f32 vector
    # multiply has no lowering (`aievec.mul_elem` is bf16-only, like the MAC), so the
    # factors go through bf16 -- `mulf` over two `extf`-from-bf16 operands is the
    # supported `mul_elem(bf16, bf16) -> f32`.
    invb = Vec{M, Float32}(Vec{M, BFloat16}(one(Vec{M, Float32}) / s))
    for j in 1:N
        e = Vec{M, Float32}(Vec{M, BFloat16}(vload(Vec{M, Float32}, y, 1, j)))
        vstore!(e * invb, y, 1, j)
    end
    return nothing
end

# --- layers: each wraps the two `@iron` launches -------------------------------------

const FLAGS = ["--alloc-scheme=basic-sequential"]

# Z = A · W, a tiled reduction over the shared (K) dimension. `A`/`W` are bf16, `Z` f32.
linear!(dA, dW, dZ) =
    @iron stack_size = 3328 flags = FLAGS for bi in 1:size(dZ, 1) ÷ TILE, oj in 1:size(dZ, 2) ÷ TILE
        @init zero_tile!(dZ)
        @reduce for kk in 1:size(dW, 1) ÷ TILE
            matmul_acc!(In(dA)[bi, kk], In(dW)[kk, oj], Out(dZ)[bi, oj])
        end
    end

# H = relu(Z + B), a tiled map; H is bf16 (feeds the next matmul).
relu_layer!(dZ, dB, dH) = @iron relu_bias!(
    In(dZ)::Tile{Float32, Tuple{TILE, TILE}},
    In(dB)::Tile{Float32, Tuple{TILE, TILE}},
    Out(dH)::Tile{BFloat16, Tuple{TILE, TILE}},
)

# Y = softmax(Z + B), a tiled map; Y is f32 (class probabilities).
softmax_layer!(dZ, dB, dY) = @iron softmax!(
    In(dZ)::Tile{Float32, Tuple{TILE, TILE}},
    In(dB)::Tile{Float32, Tuple{TILE, TILE}},
    Out(dY)::Tile{Float32, Tuple{TILE, TILE}},
)

# --- problem data (plain Julia; small values, bf16-friendly) -------------------------

const BATCH, IN, H1, H2, CLASSES = 32, 32, 32, 32, 16

Random.seed!(0)
# Small bounded weights keep the logits well-conditioned (roughly [-16, 16], no exp
# overflow) regardless of the draw; all values are exact in bf16.
mat(m, n) = Float32.(rand(-1:1, m, n)) ./ 2
W1f, W2f, W3f = mat(IN, H1), mat(H1, H2), mat(H2, CLASSES)
b1, b2, b3 = (Float32.(rand(-1:1, n)) ./ 2 for n in (H1, H2, CLASSES))
Xf = Float32.(rand(-2:2, BATCH, IN)) ./ 2

# bf16 device operands, built with comprehensions (a `BFloat16.(...)` broadcast hits an
# LLVM x86 codegen bug at width 16 -- see feedforward_relu.jl).
tobf(A) = BFloat16[BFloat16(Base.inferencebarrier(A[i, j])) for i in axes(A, 1), j in axes(A, 2)]
# Biases as full (batch, out) f32 matrices, so they co-tile with the activations.
biasmat(b, rows) = Float32[b[j] for _ in 1:rows, j in eachindex(b)]

# CPU reference: exact f32 forward pass.
relu(A) = max.(A, 0.0f0)
function softmax_rows(L)
    reduce(vcat, [(e = exp.(L[i, :] .- maximum(L[i, :])); (e ./ sum(e))') for i in axes(L, 1)])
end
logits_ref = relu(Xf * W1f .+ b1') |> H -> relu(H * W2f .+ b2') |> H -> H * W3f .+ b3'
Yref = softmax_rows(logits_ref)

if get(ENV, "IRON_RUN", "0") == "1"
    dX = NPUArray(tobf(Xf))
    dW1, dW2, dW3 = NPUArray(tobf(W1f)), NPUArray(tobf(W2f)), NPUArray(tobf(W3f))
    dB1 = NPUArray(biasmat(b1, BATCH)); dB2 = NPUArray(biasmat(b2, BATCH)); dB3 = NPUArray(biasmat(b3, BATCH))
    dZ1 = NPUArray{Float32}(undef, Tile{Float32, Tuple{BATCH, H1}})
    dZ2 = NPUArray{Float32}(undef, Tile{Float32, Tuple{BATCH, H2}})
    dZ3 = NPUArray{Float32}(undef, Tile{Float32, Tuple{BATCH, CLASSES}})
    dH1 = NPUArray{BFloat16}(undef, Tile{BFloat16, Tuple{BATCH, H1}})
    dH2 = NPUArray{BFloat16}(undef, Tile{BFloat16, Tuple{BATCH, H2}})
    dY = NPUArray{Float32}(undef, Tile{Float32, Tuple{BATCH, CLASSES}})

    linear!(dX,  dW1, dZ1); relu_layer!(dZ1, dB1, dH1)      # layer 1
    linear!(dH1, dW2, dZ2); relu_layer!(dZ2, dB2, dH2)      # layer 2
    linear!(dH2, dW3, dZ3); softmax_layer!(dZ3, dB3, dY)    # output layer

    Y = Array(dY)
    rows_ok = all(abs.(sum(Y, dims = 2) .- 1) .< 1.0f-2)
    correct = count(argmax(Y[i, :]) == argmax(Yref[i, :]) for i in 1:BATCH)
    maxerr = maximum(abs.(Y .- Yref))
    if rows_ok && correct == BATCH && maxerr < 0.05
        println("MLP softmax: PASS  (", BATCH, " samples, ", CLASSES, " classes)")
        println("  argmax matches reference: ", correct, "/", BATCH,
                ";  max |Y - Yref| = ", round(maxerr, digits = 4))
        println("  sample 1 probabilities: ", round.(Y[1, :], digits = 3))
    else
        println("MLP softmax: MISMATCH  (rows_ok=", rows_ok, ", argmax=", correct, "/", BATCH,
                ", maxerr=", round(maxerr, digits = 4), ")")
        println("  got Y[1,:]:      ", round.(Y[1, :], digits = 3))
        println("  expected Yref[1,:]: ", round.(Yref[1, :], digits = 3))
    end
else
    println("3-layer MLP classifier: Y = softmax(relu(relu(X·W1+b1)·W2+b2)·W3+b3)")
    println("  ", BATCH, "x", IN, " input, hidden ", H1, "/", H2, ", ", CLASSES, " classes")
    println("Run on an NPU with:  IRON_RUN=1 julia --project=examples examples/mlp_softmax.jl")
    println()
    println("CPU reference -- sample 1 class probabilities:")
    println("  ", round.(Yref[1, :], digits = 3), "  (argmax = class ", argmax(Yref[1, :]), ")")
    println("  all rows sum to 1: ", all(abs.(sum(Yref, dims = 2) .- 1) .< 1.0f-5))
end
