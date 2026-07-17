"""
    IRON

Program AMD NPUs from Julia, on top of MLIR-AIE.

A design is described with object FIFOs and workers, much like the Python IRON
API, but the compute kernel running on each core is an ordinary Julia function:
it is type-inferred, structurized and lowered to MLIR rather than written by hand.
The resulting `aie` module is compiled and run through MLIR-AIE's `aiecc` and XRT.

```julia
using IRON

const Buf = Tile{Int32,Tuple{1024}}

function add_one(a::Buf, b::Buf)
    for i in 1:1024
        b[i] = a[i] + Int32(1)
    end
    return nothing
end

of_in  = ObjectFifo{Buf}("in")
of_out = ObjectFifo{Buf}("out")

rt = Runtime()
start!(rt, Worker(add_one, [consumer(of_in), producer(of_out)]))
fill!(rt, producer(of_in), 1)
drain!(rt, consumer(of_out), 2)

program = Program(npu2, rt, [Buf, Buf])
println(generate_mlir(program))
```
"""
module IRON

using MLIR: IR, API
using MLIR.Dialects: arith, scf, memref
using IRStructurizer
using PythonCall

include("context.jl")
include("tile.jl")
include("dialects.jl")
include("kernel.jl")
include("design.jl")
include("runtime.jl")

export Tile
export AIEDevice, npu1, npu2
export ObjectFifo, Endpoint, producer, consumer
export Worker, Runtime, start!, drain!, Program
export generate_mlir

end # module
