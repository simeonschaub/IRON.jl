# A tiled matrix multiply, written once and compiled for several element types.
#
# The kernel is generic Julia: `matmul!` is not specialised per element type, and
# the same source lowers to i32, f16, bf16 or f32 arithmetic depending on the tile
# types it is instantiated with.
#
# Which of those the hardware will actually execute is a separate question, and a
# sharp one. `_MM_COMBOS` in aie/iron/kernels/linalg.py lists every type
# combination the accelerator multiplies:
#
#     (i8,i8) (i8,i16) (i8,i32) (i16,i16) (i16,i32) (bf16,bf16) (bf16,f32)
#
# f32 appears only as an accumulator, never as an operand -- there is no f32
# multiplier. An f32 x f32 kernel compiles, runs, and returns wrong data; the
# integer one returns the right answer. So this example runs the integer design,
# which is verified on hardware, and prints the MLIR for the rest.
#
# Run with the MLIR-AIE ironenv python so the compile/run half can find `aie`:
#   JULIA_PYTHONCALL_EXE=/path/to/mlir-aie/ironenv/bin/python julia --project examples/matmul.jl

using IRON
using BFloat16s: BFloat16
using DLFP8Types: Float8_E4M3FN

# One tile per core. A real design tiles a larger matrix over these.
const M, K, N = 8, 8, 8

"""
    matmul!(a, b, c)

`c = a * b`, accumulating in the element type of the tiles.
"""
function matmul!(
        a::Tile{T, Tuple{M, K}}, b::Tile{T, Tuple{K, N}}, c::Tile{T, Tuple{M, N}}
    ) where {T, M, K, N}
    for i in 1:M, j in 1:N
        acc = zero(T)
        for k in 1:K
            acc += a[i, k] * b[k, j]
        end
        c[i, j] = acc
    end
    return nothing
end

"""
    matmul_mixed!(a, b, c)

`c = a * b` with narrow operands and a wide accumulator.

This is the shape the accelerator is built around: it multiplies bf16 (or i8/i16)
and accumulates into something wider, so a kernel widens each element on load and
adds in the accumulator's type. `Float32(a[i, k])` is one `arith.extf`.

The same source serves any narrow input type, FP8 included -- see `fp8.jl`.
"""
function matmul_mixed!(
        a::Tile{Tin, Tuple{M, K}}, b::Tile{Tin, Tuple{K, N}}, c::Tile{Tacc, Tuple{M, N}}
    ) where {Tin, Tacc, M, K, N}
    for i in 1:M, j in 1:N
        acc = zero(Tacc)
        for k in 1:K
            acc += Tacc(a[i, k]) * Tacc(b[k, j])
        end
        c[i, j] = acc
    end
    return nothing
end

"""
    matmul_program(kernel, A, B, C)

A design streaming both operands in and the product back out.
"""
function matmul_program(kernel, ::Type{A}, ::Type{B}, ::Type{C}) where {A, B, C}
    of_a = ObjectFifo{A}("a")
    of_b = ObjectFifo{B}("b")
    of_c = ObjectFifo{C}("c")

    rt = Runtime()
    start!(rt, Worker(kernel, [consumer(of_a), consumer(of_b), producer(of_c)]))
    fill!(rt, producer(of_a), 1)
    fill!(rt, producer(of_b), 2)
    drain!(rt, consumer(of_c), 3)

    return Program(npu2, rt, [A, B, C])
end

square(::Type{T}) where {T} = Tile{T, Tuple{M, K}}

# Buffer allocation defaults to bank-aware, falling back to basic-sequential only
# when bank-aware reports failure. With three object FIFOs on one core it does not
# report failure -- it produces an allocation whose buffers overlap, and C comes
# back holding A's bytes. Every matrix multiply under programming_examples/ passes
# this same flag.
const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

if get(ENV, "IRON_RUN", "0") == "1"
    # Needs an NPU, XRT and the MLIR-AIE toolchain.
    T = Int32
    program = matmul_program(matmul!, square(T), square(T), square(T))
    compiled = IRON.compile(program; aiecc_flags = AIECC_FLAGS)

    # Asymmetric, and neither one the identity, so a transposed tile or a swapped
    # pair of operands changes the answer. A symmetric `a` with `b = I` is a trap:
    # it is invariant under exactly the mistakes worth catching.
    a = T[10i + j for i in 1:M, j in 1:K]
    b = T[i - 2j for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{T}(undef, square(T))
    IRON.run!(compiled, da, db, dc)

    result = Array(dc)
    expected = a * b
    if result == expected
        println("NPU matmul matches")
    else
        println("MISMATCH in $(count(result .!= expected)) of $(length(expected))")
        println("got:\n", result)
        println("expected:\n", expected)
    end
else
    # The same kernel source, compiled for each element type. Only the accumulate
    # is interesting; the arithmetic on `index` is subscript bookkeeping.
    function mac(mlir)
        lines = split(mlir, '\n')
        return [
            strip(replace(l, r"^\s*%\w+ = " => "")) for l in lines
                if occursin(r"arith\.(mul|add)[fi] ", l) && !occursin(": index", l)
        ]
    end

    println("one kernel, one line of Julia per element type:\n")
    for T in (Int16, Int32, Float16, BFloat16, Float32)
        mlir = generate_mlir(matmul_program(matmul!, square(T), square(T), square(T)))
        println("  ", rpad(string(T), 10), join(mac(mlir), " ; "))
    end

    function widening(mlir)
        for l in split(mlir, '\n')
            m = match(r"arith\.extf [^:]*: (.+)$", l)
            m === nothing || return "arith.extf " * m.captures[1]
        end
        return "(none)"
    end

    println("\nnarrow operands, wide accumulator -- the shape the hardware is built for:\n")
    for (Tin, Tacc) in ((BFloat16, Float32), (Float8_E4M3FN, Float32))
        mlir = generate_mlir(
            matmul_program(matmul_mixed!, square(Tin), square(Tin), square(Tacc))
        )
        println("  ", rpad("$Tin -> $Tacc", 26), widening(mlir), " ; ", join(mac(mlir), " ; "))
    end
end
