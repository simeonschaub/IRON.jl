# The vector-increment design: the host streams 1024 int32s in, a core adds one to
# each, and the result streams back. Julia port of the IRON example in test.py.
#
# Run with the MLIR-AIE ironenv python so the compile/run half can find `aie`:
#   JULIA_PYTHONCALL_EXE=/path/to/mlir-aie/ironenv/bin/python julia --project examples/add_one.jl

using IRON

const Buf = Tile{Int32, Tuple{1024}}

# The kernel: an ordinary Julia function, compiled to MLIR and inlined into the core.
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

if get(ENV, "IRON_RUN", "0") == "1"
    # Needs an NPU, XRT and the MLIR-AIE toolchain.
    compiled = IRON.compile(program)
    a = NPUArray(Int32.(0:1023))
    b = NPUArray{Int32}(undef, Buf)
    IRON.run!(compiled, a, b)
    result = Array(b)
    @assert result == Int32.(1:1024)
    println("NPU result: ", result[1:8], " ... ", result[(end - 3):end])
else
    print(generate_mlir(program))
end
