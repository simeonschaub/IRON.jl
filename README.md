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
