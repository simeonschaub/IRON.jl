# A streaming feed-forward neural network on the NPU, expressed as an `@iron` **pipeline**:
#
#     Y = softmax( relu( relu(X·W1 + b1) ·W2 + b2 ) ·W3 + b3 )
#
# Unlike examples/mlp_softmax.jl -- which runs each layer as a separate launch and bounces
# the activations through DRAM between them -- this is a single design in which every layer
# is its own compute core and the intermediate activations stream **core -> core through
# on-chip object FIFOs**, never touching DDR:
#
#     host --[X]--> core1: relu(X·W1+b1) --[A1]--> core2: relu(A1·W2+b2) --[A2]-->
#           core3: softmax(A2·W3+b3) --[Y]--> host
#
# The `@iron begin ... end` block is the pipeline form: each statement is one stage on one
# core. `In(x)`/`Out(y)` are the host boundary (streamed from/to DDR); a bare stream produced
# by one stage and consumed by the next (`dA1`, `dA2`) becomes an inter-core FIFO. The stages
# lower to a multi-worker `Program` -- see src/compiler/mlir/dataflow.jl.
#
# Everything is one 16x16 tile: 16 samples on the vector lanes, 16 features per layer, so a
# layer is a single-tile matmul the core does in one kernel call (no cross-tile reduction).
# Operands are bf16 with f32 accumulate, the mixed precision the vector MAC is built around;
# activations are emitted in bf16 so they feed the next matmul directly (see
# examples/feedforward_relu.jl and examples/mlp_softmax.jl for the bf16/exp/max details).
#
# Compiling and running need the AIE toolchain JLLs and an NPU, but no Python:
#   IRON_RUN=1 julia --project=examples examples/streaming_mlp.jl
# Without IRON_RUN the CPU reference is computed and printed, so the file runs anywhere.

using IRON
using BFloat16s: BFloat16
using Random

const TILE = 16   # AIE2 vector width: 16 f32 lanes / 32 bf16, and every dimension here.

# --- stage kernels: ordinary Julia functions, inlined into a core by `@iron` ----------

"""
    relu_layer!(x, w, b, h)

One hidden layer, fused into a single core: `h = relu(x·w + b)`, narrowed to bf16 so it
streams straight into the next layer's matmul. The matmul is column-major -- a column of the
accumulator starts from the bias, then accumulates `x[:,p] * w[p,j]` down the shared
dimension -- and the relu is a bf16 `max` (the type `aievec.max` supports), exact because the
output is bf16 anyway.
"""
function relu_layer!(
        x::Tile{BFloat16, Tuple{M, K}}, w::Tile{BFloat16, Tuple{K, N}},
        b::Tile{Float32, Tuple{M, N}}, h::Tile{BFloat16, Tuple{M, N}},
    ) where {M, K, N}
    zerob = zero(Vec{M, BFloat16})
    for j in 1:N
        acc = vload(Vec{M, Float32}, b, 1, j)             # start from the bias
        for p in 1:K
            av = vload(Vec{M, BFloat16}, x, 1, p)
            bv = Vec{M, BFloat16}(w[p, j])                # broadcast w[p, j]
            acc = muladd(Vec{M, Float32}(av), Vec{M, Float32}(bv), acc)
        end
        vstore!(max(Vec{M, BFloat16}(acc), zerob), h, 1, j)   # relu, narrow to bf16
    end
    return nothing
end

"""
    softmax_layer!(x, w, b, y)

The output layer: the logits `x·w + b` into `y` (f32), then softmax per sample (per lane)
over the `N` class columns. See examples/mlp_softmax.jl for why the max and exp are taken in
bf16 (no f32 `aievec.max`/hardware exp) while the sum is f32 and the normalise multiply goes
through bf16. `y` doubles as scratch across the three softmax passes.
"""
function softmax_layer!(
        x::Tile{BFloat16, Tuple{M, K}}, w::Tile{BFloat16, Tuple{K, N}},
        b::Tile{Float32, Tuple{M, N}}, y::Tile{Float32, Tuple{M, N}},
    ) where {M, K, N}
    # logits -> y (f32)
    for j in 1:N
        acc = vload(Vec{M, Float32}, b, 1, j)
        for p in 1:K
            av = vload(Vec{M, BFloat16}, x, 1, p)
            acc = muladd(Vec{M, Float32}(av), Vec{M, Float32}(Vec{M, BFloat16}(w[p, j])), acc)
        end
        vstore!(acc, y, 1, j)
    end
    # per-sample max over classes (bf16), for numerical stability
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
    # normalise: reciprocal (aievec.inv) then multiply, the factors through bf16
    invb = Vec{M, Float32}(Vec{M, BFloat16}(one(Vec{M, Float32}) / s))
    for j in 1:N
        e = Vec{M, Float32}(Vec{M, BFloat16}(vload(Vec{M, Float32}, y, 1, j)))
        vstore!(e * invb, y, 1, j)
    end
    return nothing
end

# --- problem data (plain Julia; small values, exact/friendly in bf16) -----------------

const BATCH, IN, H1, H2, CLASSES = TILE, TILE, TILE, TILE, TILE   # one 16x16 tile each

Random.seed!(0)
mat(m, n) = Float32.(rand(-1:1, m, n)) ./ 2      # small weights keep logits well-conditioned
W1f, W2f, W3f = mat(IN, H1), mat(H1, H2), mat(H2, CLASSES)
b1, b2, b3 = (Float32.(rand(-1:1, n)) ./ 2 for n in (H1, H2, CLASSES))
Xf = Float32.(rand(-2:2, BATCH, IN)) ./ 2

# bf16 device operands, built with comprehensions (a `BFloat16.(...)` broadcast hits an LLVM
# x86 codegen bug at width 16 -- see feedforward_relu.jl). Biases as full (batch, out) f32
# matrices so they co-tile with the activations.
tobf(A) = BFloat16[BFloat16(Base.inferencebarrier(A[i, j])) for i in axes(A, 1), j in axes(A, 2)]
biasmat(b) = Float32[b[j] for _ in 1:BATCH, j in eachindex(b)]

# CPU reference: exact f32 forward pass.
relu(A) = max.(A, 0.0f0)
function softmax_rows(L)
    reduce(vcat, [(e = exp.(L[i, :] .- maximum(L[i, :])); (e ./ sum(e))') for i in axes(L, 1)])
end
H1ref = relu(Xf * W1f .+ b1')
H2ref = relu(H1ref * W2f .+ b2')
Yref  = softmax_rows(H2ref * W3f .+ b3')

const FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    dX = NPUArray(tobf(Xf))
    dW1, dW2, dW3 = NPUArray(tobf(W1f)), NPUArray(tobf(W2f)), NPUArray(tobf(W3f))
    dB1, dB2, dB3 = NPUArray(biasmat(b1)), NPUArray(biasmat(b2)), NPUArray(biasmat(b3))
    dY = NPUArray{Float32}(undef, Tile{Float32, Tuple{BATCH, CLASSES}})

    # On-chip activation streams between the layer cores -- allocated only for their shape;
    # the pipeline never DMAs them (they live on core-to-core FIFOs).
    dA1 = NPUArray{BFloat16}(undef, Tile{BFloat16, Tuple{BATCH, H1}})
    dA2 = NPUArray{BFloat16}(undef, Tile{BFloat16, Tuple{BATCH, H2}})

    # The streaming pipeline: three chained cores, activations flowing on-chip.
    @iron flags = FLAGS stack_size = 3328 begin
        dA1 = relu_layer!(In(dX),  In(dW1), In(dB1))
        dA2 = relu_layer!(dA1,     In(dW2), In(dB2))
        Out(dY) = softmax_layer!(dA2, In(dW3), In(dB3))
    end

    Y = Array(dY)
    rows_ok = all(abs.(sum(Y, dims = 2) .- 1) .< 1.0f-2)
    correct = count(argmax(Y[i, :]) == argmax(Yref[i, :]) for i in 1:BATCH)
    maxerr = maximum(abs.(Y .- Yref))
    if rows_ok && correct == BATCH && maxerr < 0.05
        println("streaming MLP: PASS  (", BATCH, " samples, ", CLASSES, " classes, 3 chained cores)")
        println("  argmax matches reference: ", correct, "/", BATCH,
                ";  max |Y - Yref| = ", round(maxerr, digits = 4))
        println("  sample 1 probabilities: ", round.(Y[1, :], digits = 3))
    else
        println("streaming MLP: MISMATCH  (rows_ok=", rows_ok, ", argmax=", correct, "/", BATCH,
                ", maxerr=", round(maxerr, digits = 4), ")")
        println("  got Y[1,:]:         ", round.(Y[1, :], digits = 3))
        println("  expected Yref[1,:]: ", round.(Yref[1, :], digits = 3))
    end
else
    println("streaming feed-forward MLP: Y = softmax(relu(relu(X·W1+b1)·W2+b2)·W3+b3)")
    println("  ", BATCH, "x", IN, " input, hidden ", H1, "/", H2, ", ", CLASSES, " classes")
    println("  three chained cores; activations stream core->core on-chip (no DDR between layers)")
    println("Run on an NPU with:  IRON_RUN=1 julia --project=examples examples/streaming_mlp.jl")
    println()
    println("CPU reference -- sample 1 class probabilities:")
    println("  ", round.(Yref[1, :], digits = 3), "  (argmax = class ", argmax(Yref[1, :]), ")")
end
