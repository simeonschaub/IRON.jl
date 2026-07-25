# A streaming feed-forward neural network on the NPU, expressed as an `@iron` **pipeline**:
#
#     Y = softmax( relu( relu(X·W1 + b1) ·W2 + b2 ) ·W3 + b3 )
#
# Unlike examples/mlp_softmax.jl -- which runs each layer as a separate launch and bounces
# the activations through DRAM between them -- this is a single design in which every layer
# is its own compute core and the intermediate activations stream **core -> core through
# on-chip object FIFOs**, never touching DDR:
#
#     host --[Xa]--> core1: relu(Xa·Wa1) --[Ha1]--> core2: relu(Ha1·Wa2) --[Ha2]-->
#           core3: softmax(Ha2·Wa3) --[Y]--> host
#
# The `@iron begin ... end` block is the pipeline form: each statement is one stage on one
# core. `In(x)`/`Out(y)` are the host boundary; a bare stream produced by one stage and
# consumed by the next (`dHa1`, `dHa2`) becomes an inter-core FIFO. The stages lower to a
# multi-worker `Program` -- see src/compiler/mlir/dataflow.jl.
#
# ## Bias by augmentation (why there is no bias input)
#
# An AIE2 compute tile has only two input DMA channels, so each core takes exactly two
# streams -- an activation and a weight -- plus its on-chip activation link; a separate bias
# would be a third input and overrun the tile's DMA budget. So the bias is folded into the
# matmul with the standard augmentation trick (see examples/feedforward_relu.jl):
#
#     Xa = [X | 1]     (batch, in+1)      -- append a column of ones
#     Wa = [W ; bᵀ]    (in+1, out)        -- append the bias as the last row
#     Xa · Wa = X·W + 1·bᵀ = X·W + b
#
# In a *pipeline* the intermediate activations must stay augmented too, so each hidden core
# writes the ones column onto its output (`relu_layer!`'s last `vstore!`); the next core then
# reads an already-augmented `[H | 1]`.
#
# Everything is one 16-wide tile: 16 samples on the vector lanes and 16 augmented features
# (15 real + the ones column), so a layer is a single-tile matmul the core does in one kernel
# call. Operands are bf16 with f32 accumulate, the mixed precision the vector MAC is built
# around; activations are bf16 so they feed the next matmul directly (see
# examples/feedforward_relu.jl and examples/mlp_softmax.jl for the bf16/exp/max details).
#
# Compiling and running need the AIE toolchain JLLs and an NPU, but no Python:
#   IRON_RUN=1 julia --project=examples examples/streaming_mlp.jl
# Without IRON_RUN the CPU reference is computed and printed, so the file runs anywhere.

using IRON
using BFloat16s: BFloat16
using Random

const TILE = 16   # AIE2 vector width; also the augmented feature width (15 real + 1 ones).

# --- stage kernels: ordinary Julia functions, inlined into a core by `@iron` ----------

"""
    relu_layer!(xa, wa, ha)

One hidden layer, fused into a single core: `ha = [ relu(xa·wa) | 1 ]`. The matmul (`xa·wa`
over the augmented feature dimension `Ka`) already includes the bias, being `[X|1]·[W;bᵀ]`;
the relu narrows to bf16 (a bf16 `max`, exact for a bf16 output), and the final `vstore!`
writes the ones column so the activation streams out already augmented for the next core.
"""
function relu_layer!(
        xa::Tile{BFloat16, Tuple{M, Ka}}, wa::Tile{BFloat16, Tuple{Ka, N}},
        ha::Tile{BFloat16, Tuple{M, Na}},
    ) where {M, Ka, N, Na}
    zerob = zero(Vec{M, BFloat16})
    for j in 1:N
        acc = zero(Vec{M, Float32})
        for p in 1:Ka
            av = vload(Vec{M, BFloat16}, xa, 1, p)
            bv = Vec{M, BFloat16}(wa[p, j])                # broadcast wa[p, j]
            acc = muladd(Vec{M, Float32}(av), Vec{M, Float32}(bv), acc)
        end
        vstore!(max(Vec{M, BFloat16}(acc), zerob), ha, 1, j)   # relu, narrow to bf16
    end
    vstore!(one(Vec{M, BFloat16}), ha, 1, Na)   # augmentation ones column (Na = N + 1)
    return nothing
end

"""
    softmax_layer!(xa, wa, y)

The output layer: the logits `xa·wa` (= H·W + b) into `y` (f32), then softmax per sample
(per lane) over the `N` class columns -- no ones column, this is the final output. See
examples/mlp_softmax.jl for why the max and exp are bf16 (no f32 `aievec.max`/hardware exp)
while the sum is f32 and the normalise multiply goes through bf16. `y` doubles as scratch.
"""
function softmax_layer!(
        xa::Tile{BFloat16, Tuple{M, Ka}}, wa::Tile{BFloat16, Tuple{Ka, N}},
        y::Tile{Float32, Tuple{M, N}},
    ) where {M, Ka, N}
    for j in 1:N
        acc = zero(Vec{M, Float32})
        for p in 1:Ka
            av = vload(Vec{M, BFloat16}, xa, 1, p)
            acc = muladd(Vec{M, Float32}(av), Vec{M, Float32}(Vec{M, BFloat16}(wa[p, j])), acc)
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

# 15 real features per layer, augmented to 16 (the + 1 ones column carries the bias); a
# 16-sample batch and 16 output classes, so every tile is 16-wide.
const BATCH, IN, H1, H2, CLASSES = TILE, TILE - 1, TILE - 1, TILE - 1, TILE

Random.seed!(0)
mat(m, n) = Float32.(rand(-1:1, m, n)) ./ 2      # small weights keep logits well-conditioned
W1f, W2f, W3f = mat(IN, H1), mat(H1, H2), mat(H2, CLASSES)
b1, b2, b3 = (Float32.(rand(-1:1, n)) ./ 2 for n in (H1, H2, CLASSES))
Xf = Float32.(rand(-2:2, BATCH, IN)) ./ 2

# bf16 augmented operands, built with comprehensions (a `BFloat16.(...)` broadcast hits an
# LLVM x86 codegen bug at width 16 -- see feedforward_relu.jl). `augx` appends the ones
# column, `augw` appends the bias as the last row.
bf(x) = BFloat16(Base.inferencebarrier(x))
augx(X) = BFloat16[j <= size(X, 2) ? bf(X[i, j]) : one(BFloat16) for i in axes(X, 1), j in 1:size(X, 2) + 1]
augw(W, b) = BFloat16[i <= size(W, 1) ? bf(W[i, j]) : bf(b[j]) for i in 1:size(W, 1) + 1, j in axes(W, 2)]

# CPU reference: exact f32 forward pass (with the biases).
relu(A) = max.(A, 0.0f0)
function softmax_rows(L)
    reduce(vcat, [(e = exp.(L[i, :] .- maximum(L[i, :])); (e ./ sum(e))') for i in axes(L, 1)])
end
H1ref = relu(Xf * W1f .+ b1')
H2ref = relu(H1ref * W2f .+ b2')
Yref  = softmax_rows(H2ref * W3f .+ b3')

const FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    dXa = NPUArray(augx(Xf))                       # [X | 1]      (16, 16)
    dWa1, dWa2 = NPUArray(augw(W1f, b1)), NPUArray(augw(W2f, b2))   # [W ; bᵀ]  (16, 15)
    dWa3 = NPUArray(augw(W3f, b3))                 # [W3 ; b3ᵀ]   (16, 16)
    dY = NPUArray{Float32}(undef, Tile{Float32, Tuple{BATCH, CLASSES}})

    # On-chip augmented activation streams between the layer cores -- allocated only for their
    # shape; the pipeline never DMAs them (they live on core-to-core FIFOs).
    dHa1 = NPUArray{BFloat16}(undef, Tile{BFloat16, Tuple{BATCH, H1 + 1}})
    dHa2 = NPUArray{BFloat16}(undef, Tile{BFloat16, Tuple{BATCH, H2 + 1}})

    # The streaming pipeline: three chained cores, activations flowing on-chip.
    @iron flags = FLAGS stack_size = 3328 begin
        dHa1 = relu_layer!(In(dXa), In(dWa1))
        dHa2 = relu_layer!(dHa1, In(dWa2))
        Out(dY) = softmax_layer!(dHa2, In(dWa3))
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
    println("  ", BATCH, "x", IN, " input, hidden ", H1, "/", H2, ", ", CLASSES, " classes (bias by augmentation)")
    println("  three chained cores; activations stream core->core on-chip (no DDR between layers)")
    println("Run on an NPU with:  IRON_RUN=1 julia --project=examples examples/streaming_mlp.jl")
    println()
    println("CPU reference -- sample 1 class probabilities:")
    println("  ", round.(Yref[1, :], digits = 3), "  (argmax = class ", argmax(Yref[1, :]), ")")
end
