# Compiling the code *around* the kernels

*Prototype + design sketch. The core-body half is implemented and working (see "What is
implemented" below); the runtime-sequence and device-structure halves are sketched. This is
the layer underneath [stage-graph.md](stage-graph.md) — see the note at the end.*

## The idea

IRON already compiles Julia to MLIR. It just does it for one thing only: the kernel body.
Everything *around* the kernel — the core's loop nest, its acquires and releases, the host
DMA program — is built by calling op constructors by hand, which is what makes
`examples/mlp_array.jl` 466 lines.

But `compile_kernel!` is not kernel-specific. It takes any Julia function and emits its body
into any block:

```julia
compile_kernel!(ctx, block, f, argtypes, args)
```

So the question is not "can we build a nicer DSL over the op constructors" but **"can the
core body and the runtime sequence just be Julia functions too, compiled by the path that
already exists?"** If so, the abstraction problem mostly dissolves: residency, loop
structure and pipelining stop being concepts a DSL has to name and go back to being *where
you put a statement in a loop*, which is what they are in MLIR.

## What is implemented

A core body of `examples/mlp_array.jl` — the weight-stationary nested loop that took
`emit_core!` 40 lines of builder calls — written as ordinary Julia:

```julia
function mlp_core(x::Reader{:act, ActT}, w::Reader{:wt, WT}, h::Writer{:out, OutT})
    while true
        wb = acquire!(w)                 # stationary: hoisted out of the tile loop
        for _ in 1:TILES
            xt = acquire!(x)
            ht = acquire!(h)
            dense_relu!(xt, wb, ht)      # the kernel, still an ordinary call
            release!(x)
            release!(h)
        end
        release!(w)                      # so a re-launch can refill the depth-1 FIFO
    end
end

compile_kernel!(ctx, block, mlp_core,
                Tuple{Reader{:act,ActT}, Reader{:wt,WT}, Writer{:out,OutT}}, IR.Value[])
```

This compiles, and the MLIR it produces is the same module `emit_core!` builds by hand,
op for op, modulo one cosmetic difference: Julia's `for _ in 1:32` gives `scf.for %arg1 = 1
to 33` where the hand-written version says `0 to 32`.

```mlir
scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
  %0 = "aie.objectfifo.acquire"() <{objFifo_name = @wt, port = 1, size = 1}> ...
  %1 = "aie.objectfifo.subview.access"(%0) <{index = 0}> ...
  scf.for %arg1 = %c1 to %c33 step %c1 {
    %2 = "aie.objectfifo.acquire"() <{objFifo_name = @act, port = 1, size = 1}> ...
    ...                                          // dense_relu! inlined here
    "aie.objectfifo.release"() <{objFifo_name = @act, port = 1, size = 1}> ...
```

Note what is *not* in the Julia source: `Stationary`, `Streaming`, `pipeline`, `depth`.
Weight-stationarity is the acquire sitting outside the inner loop. Re-launch safety is the
release sitting inside the outer one. There is nothing to name because the loop nest already
says it.

### The pieces it needed

Four gaps, each small, and the third is the interesting one.

**1. `FifoRef{Name, Port, T}`** (`src/device/fifo.jl`). `aie.objectfifo.acquire` names its
FIFO with a *symbol attribute*, not an operand, so the name has to be a compile-time
constant. Putting it in a type parameter makes it one that inference can see, and makes a
core body's *signature* the record of which FIFOs it is wired to. A `FifoRef` is a
singleton: it lowers to no MLIR value at all, so `compile_kernel!` now skips singleton
parameters when binding `args` positionally.

**2. `while true`.** IRStructurizer reports it as a `LoopOp` (not a `WhileOp` — that is for
condition-at-header loops), which the emitter did not handle, and its return type is
`Union{}`, which `compile_kernel!` rejected. `emit_loop!` now emits `scf.for 0:typemax(Int)`
— the idiom the hand-written designs already use — and `Union{}` is accepted, since a core
that never returns is exactly what a core is.

**3. `Tile` had to become `mutable`.** This is the one that is a modelling bug rather than a
missing feature. An immutable struct with no fields is a *singleton type*, so inference
knows the value of anything typed `Tile{T,Dims}` exactly — and duly constant-folded
`acquire!`'s result into a `QuoteNode`, severing the SSA edge from the acquire to the
kernel that reads it:

```julia
%2 = invoke acquire!(x)::Tile{BFloat16, Tuple{16, 256}}   # fine
     invoke kern!($(QuoteNode(Tile{BFloat16, Tuple{16, 256}}())), ...)   # not fine
```

No `inferencebarrier` can fix this: inference is *right*, there is only one value of a
singleton type. A tile is a reference to a buffer and must have identity, so `Tile` is now
`mutable struct`. It never mattered before only because tiles arrived as arguments and were
never the result of a call.

**4. Nested calls.** `dense_relu!` is too big for Julia's inliner, so it survives as a call
in the core body's IR. Rather than force `@inline` on every kernel, `emit_call!` now falls
back to compiling the callee's body inline at the call site, binding its parameters to the
operands. This is the seam that makes the whole thing work: the kernel stays a call in the
source and stops being one only in the emitter.

### `widen_call_result`

Inference widens the result type of a call whose value is unused to `Any` — locally nothing
needs it, so computing it precisely is wasted work. A kernel compiler *is* downstream of
inference and does need it: a core body is almost entirely calls made for their effects
(`release!`, a kernel invoked on tiles it writes), and every one of them has an unused
result, so they all arrived typed `Any`.

Julia 1.14-DEV exposes a hook for exactly this, and `src/compiler/interpreter.jl` now
overloads it:

```julia
CC.widen_call_result(::IRONInterpreter, ::CC.StmtInfo, ::CC.CallInferenceState,
                     ::CC.AbsIntState) = false
```

Verified in isolation on nightly: the unused calls go from `Any` to `Core.Const(nothing)`.
It is `@static`-guarded because the hook does not exist in 1.12 (there the widening is
inline in `abstract_call_gf_by_type`, with nothing to overload), and IRON cannot currently
load on nightly, so it is untested *in IRON*. The emitter therefore does not depend on it:
it gates nested-call inlining on the use index, which is the honest question anyway ("does
anything need a value from this call?") and cannot be widened away.

## What is sketched, not built

### The runtime sequence

Same treatment, with `dma!`/`await!` intrinsics over `aiex.dma_configure_task_for` and
friends:

```julia
function sequence(X::HostBuf{...}, W::NTuple{4,HostBuf{...}}, Y::HostBuf{...})
    for s in 1:4
        await!(dma!(W[s] => Weights[s]))          # phase 1: params, once
    end
    for t in 0:(NTILES - 1)                       # phase 2: batch tiles
        dma!(view(X, t), Act[t % PIPES + 1])
        dma!(Out[t % PIPES + 1], view(Y, t))
    end
end
```

The payoff here is bigger than tidiness. Today `emit_runtime!` *unrolls* the tile loop — 64
tiles becomes 64 pairs of hand-emitted tasks, and `mlp_array.jl` compiles to 5580
instruction words largely because of it. A compiled `for` stays a `for`, which needs the BD
offset/len to be an SSA value rather than an attribute — and that is precisely what
mlir-aie #3306 added (`aie.dma_bd` via `DynamicIndexList`). `aie.jl` notes that IRON "only
ever needs static values" and puts them in `static_*`; a compiled runtime sequence is the
reason to start using the dynamic path.

### The device structure

`aie.device`'s body — tiles, FIFOs, links — is declarative, not imperative, so there is
nothing to compile. It wants a *builder* with an implicit insertion point, the way MLIR's
own Python bindings work, so that ordinary Julia (`for p in 1:PIPES, s in 1:NSTAGES`) is the
metaprogramming layer:

```julia
@device npu2 "mlp" begin
    for p in 1:PIPES, s in 1:NSTAGES
        mem = memtile!(); cores = [coretile!() for _ in 1:PAR]
        fifo!(:"w_l3l2_p$(p)_s$s", shim => mem, layer_tile(s), depth = 1)
        link!([:"w_l3l2_p$(p)_s$s"] => [...]; dst_offsets = ...)
        core!(cores[j], mlp_core, stack_size = 3072)
    end
end
```

This is the part where Julia is the macro language for MLIR rather than a compiled one, and
the split is clean: **trace time builds the structure, compile time fills the regions.**

Two open questions in that seam. Unrolling *inside* a compiled body has no trace time to
happen at, so it needs an `@unroll` of its own. And a body's FIFO wiring currently lives in
its argument types, which means the `@device` block has to spell out the same names twice —
worth a `core!` that derives the signature from the FIFOs it is given.

## Relation to the stage graph

[stage-graph.md](stage-graph.md) proposes a `StageGraph`/`stage!` API that derives FIFOs,
links and offsets from a `split = :N => 4` declaration. Two of the three gaps it was written
to close are better closed here:

* **Operand residency** — the stage graph invented `param!` vs `source!` to say "stationary".
  Here it is just where the acquire sits. Delete the concept.
* **Stage axis** — a stage graph node was going to carry a loop nest it never spelled out.
  Here you write the loop nest.

What the stage graph still buys, and this layer does not, is the *derivation*: turning
`split = :N => 4` into four FIFOs plus a link with the right `src_offsets`, and checking the
result against the 2-input-DMA-channels-per-core and 6-channel-per-MemTile budgets. That is
real work and worth having. But it should generate *this* — Julia core bodies and a `@device`
block — rather than op constructors, and it should be a library on top rather than the base
layer. Build the base first.
