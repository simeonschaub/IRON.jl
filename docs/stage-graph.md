# A stage-graph API for IRON

*Design sketch. Nothing here is implemented; `examples/mlp_array.jl` is the hand-written
design this is trying to replace, and every rule below is stated as "what would have to be
emitted to reproduce that file".*

> **Superseded in part by [julia-as-mlir.md](julia-as-mlir.md)**, which compiles the core
> body and the runtime sequence from ordinary Julia. That removes the need for two of the
> three concepts below — operand residency and the stage axis both become "where you put
> the statement in the loop nest". What survives is the *derivation* (a `split` into FIFOs,
> links and offsets) and the *resource model*, and both should target that layer rather
> than the op constructors. Read that one first.

## What's missing

`examples/mlp_array.jl` is 466 lines of hand-assembled `aie` ops for a 4-layer MLP on 32
cores. Almost none of it is *choices* — it is the mechanical consequence of four decisions
(4 layers, split output features 4 ways, weights stationary, 2 pipeline copies). `@iron`
cannot express it, for three structural reasons:

1. **No operand residency.** A `Worker` acquires *and releases* every endpoint on every
   iteration, so every operand is streaming. Weight-stationarity — the thing that makes an
   NN compute-bound rather than DMA-bound — is a *lifetime* property of an operand, and
   there is nowhere to write it.
2. **No stage axis.** `@iron begin…end` gives many stages, one core each. `@iron for` +
   `@cores` gives many cores, one stage. This design is the product: stage × feature-split ×
   pipeline-copy. Neither surface form has that iteration space.
3. **L2 belongs to an operand, not an edge.** `@iron`'s `L2(broadcast/distribute/join)` is
   defined relative to one kernel's operands and the reduction's `@cores` axis. Here a
   MemTile is simultaneously stage *s*'s output collector *and* stage *s+1*'s input
   distributor. It belongs to the edge between two stages, and there is no edge.

So the proposal is not more sugar over `Worker`. It's a different model: **a graph of
stages, where each stage declares a kernel, a replication, and the residency of each
operand — and every FIFO, link, offset and DMA is derived from that.**

## The model

A stage graph has three kinds of node and one kind of edge.

| | | |
|---|---|---|
| `source!` / `sink!` | host boundary | one tile per step, streamed from/to DDR |
| `param!` | launch-invariant data | loaded once per launch, resident in L1 |
| `stage!` | a replicated kernel | `replicas` cores, one column, one MemTile |
| edge | a stream between two of the above | materialised in L2 iff the two ends disagree about its layout |

Three properties are orthogonal and each is declared exactly once:

* **residency** — `param!` is stationary by construction; everything else streams. (See
  *Open questions* for `state!`, the loop-carried case.)
* **partitioning** — a stage says `split = :N => 4`, naming an *axis of the kernel's type
  signature*, not a dimension number.
* **replication of the whole graph** — `pipelines`, a launch-level property, because it is
  pure resource arithmetic.

### Splitting by named axis

This is the load-bearing idea, and IRON already has the hook for it. Kernels are written
with their shapes as type variables:

```julia
dense_relu!(x::Tile{BFloat16,Tuple{M,K}}, wb::Tile{BFloat16,Tuple{K1,N}},
            h::Tile{BFloat16,Tuple{M,N}}) where {M,K,K1,N}
```

`split = :N => 4` then means: cut the `N` axis four ways, and for each operand look up where
`N` appears in its signature.

| operand | signature | `N` at | distribution |
|---|---|---|---|
| `x` | `Tuple{M,K}` | — | **broadcast** — every replica needs the whole thing |
| `wb` | `Tuple{K1,N}` | dim 2 | **split** — replica *j* gets columns *j*-block |
| `h` | `Tuple{M,N}` | dim 2 | **split** — replica *j* produces columns *j*-block |

Nothing else has to be said. Broadcast-vs-split, the slice shapes, the link offsets and the
need for a join all fall out of that one table. It is the same move `@iron` already makes
("a tile shape is inferred from the buffer and the extents of the axes indexing it"), lifted
from loop variables to operand shapes.

## The API

```julia
struct StageGraph
    tiles::Int              # batch tiles the host feeds per launch
    nodes::Vector{Node}
    edges::Vector{Edge}
end

source!(g, ::Type{<:Tile})                  -> Stream    # from host, one tile per step
sink!(g, s::Stream)                         -> nothing   # to host
param!(g, ::Type{<:Tile})                   -> Stream    # stationary, one DMA per launch
stage!(g, kernel, ins...; out, split, name) -> Stream    # a replicated kernel

IRON.compile(g; device = npu2, pipelines = :fill, flags, stack_size) -> CompiledProgram
```

* `split = :axis => n`, or `nothing` for a single core.
* `pipelines = :fill` replicates the whole graph as many times as the array has columns for,
  and round-robins batch tiles across the copies; an integer pins it.
* Host buffers are passed to `run!` in declaration order: sources, params, sinks.

### The MLP, rewritten

The whole of `examples/mlp_array.jl`, minus the kernels (which are unchanged) and the host
reference:

```julia
g = StageGraph(tiles = NTILES)

x = source!(g, Tile{BFloat16, Tuple{BATCH, WIDTH}})

for (s, (K, N)) in enumerate(SHAPES)
    kernel = s == NSTAGES ? dense_logits! : dense_relu!
    T      = s == NSTAGES ? Float32 : BFloat16
    w = param!(g, Tile{BFloat16, Tuple{K + 1, N}})       # [W ; bᵀ], held in L1
    x = stage!(g, kernel, x, w;
               out   = Tile{T, Tuple{BATCH, N}},
               split = :N => PAR)
end

sink!(g, x)

prog = IRON.compile(g; device = npu2, pipelines = :fill)
run!(prog, X, W1, W2, W3, W4, Y)
```

Twelve lines. The four decisions are the four things visible: the loop over `SHAPES`, the
`param!`, the `split`, the `pipelines`.

Optional sugar, in the spirit of `@iron begin…end`'s pipeline form — worth having, but a
distant second to getting the model right:

```julia
@stages tiles = NTILES pipelines = :fill begin
    h1     = dense_relu!(In(X), Param(W1))   split :N => 4
    h2     = dense_relu!(h1,    Param(W2))   split :N => 4
    h3     = dense_relu!(h2,    Param(W3))   split :N => 4
    Out(Y) = dense_logits!(h3,  Param(W4))   split :N => 4
end
```

## Derivation rules

What the compiler emits, stated against the ops `examples/mlp_array.jl` writes by hand.
`R` = replicas, `D` = FIFO depth (2 streaming, 1 stationary).

**`param!` consumed by a stage with `split = :N => R`.** One FIFO shim → MemTile carrying
the *whole* matrix at depth 1 (`w_l3l2_p{p}_s{s}`); `R` FIFOs MemTile → core at depth 1
(`w_l2l1_…_c{j}`); one `aie.objectfifo.link` with
`dst_offsets[j] = (j-1) * prod(slice_extents)` — a **distribute**.

**A stage output that is split, consumed by an operand that is broadcast.** The two ends
disagree, so the edge is materialised in the producer's MemTile: `R` FIFOs core → MemTile
(`out_…_c{j}`), one link joining them with `src_offsets[j] = (j-1) * prod(slice_extents)`,
feeding a single FIFO MemTile → *all `R` consumer cores* (a broadcast is one FIFO with
several consumer operands). This is the join+broadcast MemTile, derived rather than written.

> **Rule:** an edge needs an L2 relay iff its producer distribution ≠ its consumer
> distribution. Equal distributions on both ends are a direct core → core FIFO, which is
> what `streaming_mlp.jl` already gets from `@iron`.

**`source!`** — one FIFO shim → all consumer cores, depth 2. **`sink!`** — the join above,
into a FIFO MemTile → shim.

**Residency, in the core body.** This is the nested loop `emit_core!` builds by hand:

```
forever                                  # one iteration per launch
    acquire every stationary operand     # blocks on the host's one-off DMA
    for _ in 1:tiles_per_copy            # the batch this launch feeds
        acquire streaming operands; kernel(...); release them
    end
    release every stationary operand     # so the next launch can refill
```

The stationary acquire is hoisted out of the batch loop (that *is* stationarity) but stays
inside the per-launch loop, or a re-launch deadlocks on the depth-1 param FIFO — a bug worth
paying an abstraction to never write again.

**The runtime sequence.** Entirely derivable: phase 1 issues one contiguous DMA per param
and awaits them; phase 2 loops over batch tiles, computing each source/sink transfer with
`_tile_pattern`, assigning tile *t* to copy `t % pipelines`, and keeping a depth-deep sliding
window of in-flight tasks per FIFO so drains overlap the next tile's compute. Both
`emit_runtime!` and `_emit_schedule_runtime!` already implement this window; it should be
factored out once and shared.

**Offsets are a plain cut only when the split axis is trailing.** A `Tile` is column-major,
so slicing its *last* Julia dimension is a contiguous cut and the link needs only an offset.
Splitting an interior axis is a strided pattern and the relay needs `dims_to_stream`. The
API should either reject the interior case with that explanation or emit the pattern — not
silently produce the wrong offsets.

## The resource model

The other half of the job, and the half a nicer syntax alone wouldn't buy. These budgets
decide whether a design places at all; today you find out from the placer. The API should
compute them at graph-construction time and print a table.

Per core (one replica):

```
in_channels  = #streaming inputs + #params              ≤ 2
out_channels = #outputs                                 ≤ 2
L1 bytes     = Σ depth × sizeof(slice) + stack_size     ≤ 64 KB
```

For the MLP's stage 1: 2×(16×256×2) act + 1×(257×64×2) weights + 2×(16×64×2) out + 3072 B
stack = 56,448 B, against the 56,460 B the real build reports (the dozen bytes are the
allocator's, not the model's) and a 65,536 B ceiling.

That `in_channels ≤ 2` is why the bias rides in the weight slab as a `K+1`th row rather than
arriving as a third stream — the API should say that in an error message.

Per MemTile, with `R` replicas:

```
S2MM = 1 (param in) + R (join)      ≤ 6
MM2S = R (distribute) + 1 (bcast)   ≤ 6
bytes                               ≤ 512 KB
```

So **`R ≤ 5`**, and `PAR = 4` in the example is the power of two under the hard ceiling —
not a taste call. Worth having the compiler be able to say so.

Per copy, shim channels (2 MM2S + 2 S2MM each): `1 + nstages` MM2S for the input and the
params, 1 S2MM for the output → `ceil((1 + nstages)/2)` = 3 shims for the 4-layer MLP, ×2
copies = 6 of 8. That arithmetic is the hand-written `wshim(p, s)` in the example, and
allocating shims is exactly the sort of thing the user should never do.

**Placement.** A stage's replicas must share a MemTile, so they must land in one column, and
consecutive stages should be adjacent. Today `logical_tile` + `--aie-place-tiles` places
freely and happens to do the right thing; the API needs to express the grouping constraint
rather than rely on that.

## What it doesn't cover

* **Splitting the reduction axis.** Only `:N` (output features) splits here, so a stage's
  whole `K` must fit one core's weight slab. At `WIDTH = 256` that's 33 KB and fine; at
  `WIDTH = 1024` it's 525 KB and impossible. Real nets need `split = :K => R` with each
  replica producing a *partial* sum and a cross-core reduction on the edge — a third
  distribution, `Partial()`, whose relay is an accumulating join rather than a concatenating
  one. This is the single most important extension, and it is what makes the stage graph
  subsume `@iron for`'s reduction rather than sit beside it.
* **Cross-replica ops.** Softmax needs a max and a sum over all classes; a replica owns
  `N/R` of them. Same missing primitive as above.
* **`state!`** — an operand that is stationary *and* written, i.e. loop-carried on-chip
  state. RNN state, a KV cache, or online normalisation statistics. Residency is already in
  the model, so this is mostly about lock discipline.
* **Fusion.** More stages than columns should map several stages onto one core, sequentially.
  The graph makes this expressible; nothing here does it.
* **Non-uniform tile rates** — pooling, striding, anything where one input tile is not one
  output tile.

## Implementation path

Roughly in dependency order; most of the pieces exist in some form.

1. Factor the runtime tile loop + sliding window out of `emit_runtime!` /
   `_emit_schedule_runtime!` into one reusable emitter.
2. Extend `dataflow.jl` with MemTile endpoints and `objectfifo.link` (today it is shim ↔
   core only, one shim per transfer, no links, whole-tile contiguous DMA). This is the
   biggest gap and everything else sits on it.
3. Add residency to `Worker` and emit the outer/inner loop nest instead of a flat body.
4. `StageGraph`, `stage!`, and the axis-matching that turns `split = :N => R` into a
   per-operand distribution.
5. The resource model and its error messages.
6. Placement grouping.
7. `Partial()` and cross-replica reduction.
8. `@stages` sugar.

Steps 1–5 are enough to compile `examples/mlp_array.jl` from the twelve-line version, which
is the obvious acceptance test: build both, diff the generated MLIR.
