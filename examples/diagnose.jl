# A bisect for a design that compiles and runs but returns wrong data.
#
# Each stage adds exactly one thing to the one before it, so the first failure
# names the culprit. Stage 1 is the design that mirrors the reference generic.mlir
# op-for-op, so if it fails the problem is in the core/runtime scaffolding rather
# than in anything the later stages add.
#
#   1. copy, 1-D, i32, 2 FIFOs   -- the reference shape
#   2. copy, 2-D, f32, 2 FIFOs   -- adds a 2-D memref and a float element type
#   3. copy, 2-D, f32, 3 FIFOs   -- adds a second input FIFO, unread by the kernel
#   4. matmul, 2-D, f32, 3 FIFOs -- adds the arithmetic
#
# Run against real hardware:
#   JULIA_PYTHONCALL_EXE=/path/to/mlir-aie/ironenv/bin/python julia --project examples/diagnose.jl

using IRON

const N1 = 1024
const M = 8

# `--aie-assign-buffer-addresses` defaults to bank-aware allocation, falling back
# to basic-sequential only when bank-aware reports failure. For a design with
# several object FIFOs on one core it does not report failure -- it just produces
# an allocation whose buffers overlap, and the output tile comes back holding an
# input's bytes. Every matrix multiply under programming_examples/ forces the other
# scheme; see programming_examples/basic/matrix_multiplication/single_core.
const BUFFER_ALLOC_FIX = ["--alloc-scheme=basic-sequential"]

copy_1d!(a::Tile{Int32, Tuple{N1}}, c::Tile{Int32, Tuple{N1}}) = begin
    for i in 1:N1
        c[i] = a[i]
    end
    nothing
end

copy_2d!(a::Tile{Float32, Tuple{M, M}}, c::Tile{Float32, Tuple{M, M}}) = begin
    for i in 1:M, j in 1:M
        c[i, j] = a[i, j]
    end
    nothing
end

# `b` is acquired and released by the worker but never read, which isolates the
# cost of a third FIFO from the arithmetic that stage 4 adds.
copy_2d_3fifo!(
    a::Tile{Float32, Tuple{M, M}}, b::Tile{Float32, Tuple{M, M}},
    c::Tile{Float32, Tuple{M, M}},
) = begin
    for i in 1:M, j in 1:M
        c[i, j] = a[i, j]
    end
    nothing
end

matmul!(
    a::Tile{Float32, Tuple{M, M}}, b::Tile{Float32, Tuple{M, M}},
    c::Tile{Float32, Tuple{M, M}},
) = begin
    for i in 1:M, j in 1:M
        acc = zero(Float32)
        for k in 1:M
            acc += a[i, k] * b[k, j]
        end
        c[i, j] = acc
    end
    nothing
end

# Build a design whose kernel reads `length(inputs)` tiles and writes one.
function design(kernel, inputs::Vector{DataType}, out::DataType)
    fifos = [ObjectFifo{T}("in$i") for (i, T) in enumerate(inputs)]
    of_out = ObjectFifo{out}("out")

    rt = Runtime()
    start!(rt, Worker(kernel, [map(consumer, fifos); producer(of_out)]))
    for (i, f) in enumerate(fifos)
        fill!(rt, producer(f), i)
    end
    drain!(rt, consumer(of_out), length(fifos) + 1)

    return Program(npu2, rt, [inputs; out])
end

function stage(name, kernel, inputs, out, hosts, expected; aiecc_flags = String[])
    program = design(kernel, inputs, out)
    mlir = generate_mlir(program)   # verifies as it goes

    if get(ENV, "IRON_RUN", "0") != "1"
        println(rpad(name, 34), "MLIR ok (", count("aie.objectfifo\"", mlir), " FIFOs)")
        return true
    end

    buffers = [IRON.device_array(h) for h in hosts]
    dc = IRON.device_zeros(out)
    IRON.run!(IRON.compile(program; aiecc_flags), buffers..., dc)

    got = IRON.host_array(dc)
    ok = got == expected
    println(rpad(name, 34), ok ? "PASS" : "FAIL")
    if !ok
        wrong = count(got .!= expected)
        println("    $wrong of $(length(expected)) elements wrong")
        println("    got:      ", vec(got)[1:min(8, end)], " ...")
        println("    expected: ", vec(expected)[1:min(8, end)], " ...")
    end
    return ok
end

const Vec32 = Tile{Int32, Tuple{N1}}
const Mat = Tile{Float32, Tuple{M, M}}

a1 = Int32.(1:N1)
a2 = Float32[10i + j for i in 1:M, j in 1:M]
b2 = Float32[i - 2j for i in 1:M, j in 1:M]

# Each stage runs even if an earlier one failed: knowing whether 2 and 3 also fail
# says whether the cause is shared or specific.
stage("1. copy 1-D i32, 2 FIFOs", copy_1d!, DataType[Vec32], Vec32, [a1], a1)
stage("2. copy 2-D f32, 2 FIFOs", copy_2d!, DataType[Mat], Mat, [a2], a2)
stage("3. copy 2-D f32, 3 FIFOs", copy_2d_3fifo!, DataType[Mat, Mat], Mat, [a2, b2], a2)
stage("4. matmul, default alloc", matmul!, DataType[Mat, Mat], Mat, [a2, b2], a2 * b2)
stage(
    "5. matmul, basic-sequential", matmul!, DataType[Mat, Mat], Mat, [a2, b2], a2 * b2;
    aiecc_flags = BUFFER_ALLOC_FIX,
)
