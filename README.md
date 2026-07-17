# IRON

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://simeonschaub.github.io/IRON.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://simeonschaub.github.io/IRON.jl/dev/)
[![Build Status](https://github.com/simeonschaub/IRON.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/simeonschaub/IRON.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/simeonschaub/IRON.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/simeonschaub/IRON.jl)

Program AMD NPUs from Julia, on top of [MLIR-AIE](https://github.com/Xilinx/mlir-aie).

A design is described with object FIFOs and workers, following the structure of
MLIR-AIE's Python IRON API. The difference is the compute kernel: instead of
writing it in a tracing DSL or in C++, you write an ordinary Julia function.
It is type-inferred, structurized with
[IRStructurizer](https://github.com/simeonschaub/IRStructurizer.jl) and lowered to
`scf`/`arith`/`memref` through [MLIR.jl](https://github.com/JuliaLabs/MLIR.jl),
then spliced into the `aie.core` body.

```julia
using IRON

const Buf = Tile{Int32, Tuple{1024}}

# Runs on a compute tile. An ordinary Julia function, compiled to MLIR.
function add_one(a::Buf, b::Buf)
    for i in 1:1024
        b[i] = a[i] + Int32(1)
    end
    return nothing
end

of_in = ObjectFifo{Buf}("in")
of_out = ObjectFifo{Buf}("out")

rt = Runtime()
start!(rt, Worker(add_one, [consumer(of_in), producer(of_out)]))
fill!(rt, producer(of_in), 1)    # host buffer 1 -> of_in
drain!(rt, consumer(of_out), 2)  # of_out -> host buffer 2

program = Program(npu2, rt, [Buf, Buf])
println(generate_mlir(program))
```

See `examples/add_one.jl` for the same design end to end. It is the Julia port of
the Python IRON vector-increment example, and produces the module in
`generic.mlir`.

## Running on hardware

Generating MLIR needs nothing but Julia. Compiling and running additionally need
the MLIR-AIE toolchain, an NPU and XRT, all of which are reached through the
Python stack via PythonCall -- there is nothing to gain from reimplementing
`aiecc` or XRT buffer management in Julia. Point PythonCall at the ironenv
interpreter before loading IRON:

```julia
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/mlir-aie/ironenv/bin/python"
using IRON

compiled = IRON.compile(program)          # writes .mlir, hands it to aiecc
a = IRON.device_array(Int32.(0:1023))     # XRT buffers on the NPU
b = IRON.device_zeros(Buf)
IRON.run!(compiled, a, b)
IRON.host_array(b) == Int32.(1:1024)
```

Keyword arguments to `compile` are forwarded to `aie.iron.jit`. One of them is
worth knowing about before it costs you a day:

```julia
compiled = IRON.compile(program; aiecc_flags = ["--alloc-scheme=basic-sequential"])
```

Buffer allocation defaults to bank-aware, falling back to basic-sequential only
when bank-aware *reports* failure. For a design with several object FIFOs on one
core it does not report failure -- it produces an allocation whose buffers
overlap, and the design runs, and returns an output tile holding an input's
bytes. Every matrix multiply under `programming_examples/` passes this flag, as
does `examples/matmul.jl`. If a design compiles and runs but returns data that
looks like one of its inputs, try this first; `examples/diagnose.jl` runs the
same matmul with and without it.

## How kernels are compiled

`Tile{T,Dims}` is an empty marker type standing for a `memref<...>`. It is never
constructed: it exists so a kernel can be inferred against it, and its
`getindex`/`setindex!` are markers the compiler rewrites into `memref.load` and
`memref.store`. Those two methods are `@noinline` and route their result through
`Base.inferencebarrier`, which matters more than it looks -- a body returning a
literal lets inference fold the load to a constant and then fold away the
arithmetic that reads it, quietly leaving an empty kernel.

Indexing is 1-based, as everywhere else in Julia; the compiler subtracts the
offset when lowering to the 0-based memref, and memoizes the adjustment so that
`a[i]` and `b[i]` share one `arith.subi`.

Supported inside a kernel: `for` loops over ranges, `if`/`else`, and integer and
floating-point arithmetic on tile elements (see `ARITH_OPS` in `src/kernel.jl`).
Kernels must return `nothing` and communicate through the tiles they write. A
call with no registered lowering is an error rather than a silently wrong kernel.

## Element types

A kernel is generic in its element type: `examples/matmul.jl` writes one
`matmul!` and compiles it for each of

| Julia | MLIR | multiply-accumulate |
|---|---|---|
| `Int32` | `i32` | `arith.muli` / `arith.addi` |
| `Float16` | `f16` | `arith.mulf` / `arith.addf` |
| `BFloat16` | `bf16` | `arith.mulf` / `arith.addf` |
| `Float32` | `f32` | `arith.mulf` / `arith.addf` |
| `Float8_E4M3FN`, `Float8_E5M2` | `f8E4M3FN`, `f8E5M2` | storage only -- see below |

The first four need nothing special: Julia lowers their arithmetic to intrinsics
already, `BFloat16` included, because LLVM has all four natively.

### What the hardware will actually run

Generating correct MLIR for a type is not the same as the core executing it, and
the gap is wide enough to be worth stating. `_MM_COMBOS` in
`aie/iron/kernels/linalg.py` lists every type combination the accelerator
multiplies -- `(i8,i8) (i8,i16) (i8,i32) (i16,i16) (i16,i32) (bf16,bf16)
(bf16,f32)` -- and f32 appears only as an accumulator, never as an operand.

Measured on an NPU2, with `examples/diagnose.jl`:

| kernel | result |
|---|---|
| copy, i32 and f32, 1-D and 2-D | passes |
| matmul, i32 | passes |
| matmul, f32 | **wrong data** |
| matmul, bf16 operands, f32 accumulator | **wrong data** |

Moving floats is fine; multiplying them scalar-wise on the core is not, and it
fails silently rather than refusing to compile. The bf16 case fails for the same
reason -- widening on load is one `arith.extf`, but the accumulate is still scalar
f32. So a scalar kernel that computes should use integers, which is why
`examples/matmul.jl` runs the i32 design.

Float throughput lives in the vector unit, and that is reachable -- see below.

## Vector kernels

`aiecc` runs `convert-vector-to-aievec` over every AIE2/AIE2p core, and that
pipeline "ingests arbitrary MLIR Vector code". So the way to the vector unit is
the standard `vector` dialect; nothing here emits `aievec` directly:

```
vector.fma  ->  aievec.mac_elem  ->  the MAC intrinsic aie::mmul uses
```

A kernel says that with `SIMD.Vec{N,T}` (`examples/matmul_vectorized.jl`):

```julia
for i in 1:M
    acc = zero(Vec{N,T})                                     # stays in a register
    for k in 1:K
        acc = muladd(Vec{N,T}(a[i,k]),                       # vector.broadcast
                     vload(Vec{N,T}, b, k, 1), acc)          # vector.load + vector.fma
    end
    vstore!(acc, c, i, 1)                                    # vector.store
end
```

**The vector width is the hardware's, not the matrix's.** `convert-vector-to-aievec`
lowers `vector.fma` only for f32 at 16 lanes, and bf16 at 16 or 32 -- AIE2's vector
registers are 512 bits. A `vector<8xf32>` matches no pattern, and aiecc stops with
`failed to legalize operation 'vector.fma'`. `vector.fma` is also floating-point
only, so an integer multiply-accumulate lowers to `arith.muli` + `arith.addi` over
vectors instead.

SIMD.jl's arithmetic would normally inline to `llvmcall` carrying a literal LLVM IR
string, which means nothing here, so the overlay method table redirects the
operators to intrinsics before that happens -- the same mechanism as the FP8
conversions.

The FP8 formats are different, in a way worth knowing about. Their Julia
packages implement arithmetic and conversion in software, since a CPU has no
FP8, so inferring a kernel against those methods buries one hardware conversion
under a few hundred integer ops. IRON therefore infers kernels under its own
`AbstractInterpreter` with an overlay method table (`src/interpreter.jl`) that
replaces just the FP8 conversions with an intrinsic, leaving everything else
alone. `Float32(a[i, k])` then emits one `arith.extf`.

FP8 arithmetic itself is deliberately not overlaid. The hardware computes in f16
or f32 and uses FP8 as a storage format, which is the split cuTile makes too, so
a kernel converts on load and accumulates in a wider type:

```julia
function matmul_fp8!(a::Tile{Float8_E4M3FN,...}, b::..., c::Tile{Float32,...})
    acc = zero(Float32)
    for k in 1:K
        acc += Float32(a[i, k]) * Float32(b[k, j])   # arith.extf, then f32 math
    end
end
```

FP8 comes from either [DLFP8Types](https://github.com/chengchingwen/DLFP8Types.jl)
or [Microfloats](https://github.com/JuliaMath/Microfloats.jl), attached through
package extensions in `ext/`. Add an element type by overloading `mlir_eltype`,
and `bitwidth` too if it is sub-byte. Formats MLIR 18 lacks (the FNUZ variants,
`Float8_E8M0FNU`, `Float6_*`, `Float4_*`) raise an unsupported-type error rather
than being mapped onto a neighbouring format.

## The aie dialect

MLIR.jl links against upstream MLIR, which has no `aie`/`aiex` dialect. Rather
than build bindings for them, the context allows unregistered dialects and the
ops are constructed generically in `src/dialects.jl`. MLIR keeps an opaque
properties slot on unregistered ops, so `<{...}>` inherent attributes, and
`#aie<...>` attributes and `!aie.<...>` types, all round-trip through the textual
form for `aie-opt` to parse with the real dialects registered.

Two details are load-bearing. Inherent attributes must be set with
`mlirOperationSetInherentAttributeByName` rather than passed as discardable
attributes: the generic parser only moves `<{...}>` into an op's properties, so a
misplaced attribute reappears as "missing" at verification. And the upstream ops
print with their custom assembly while the `aie` ops print generically, since
this context lacks the dialect; `aie-opt` accepts either.
