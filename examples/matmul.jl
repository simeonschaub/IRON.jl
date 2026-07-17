# A tiled matrix multiply, written once and compiled for several element types.
#
# The kernel is generic Julia: `matmul!` is not specialised per element type, and
# the same source lowers to i32, f16, bf16 or f32 arithmetic depending on the tile
# types it is instantiated with. `matmul_fp8!` shows the split the hardware
# actually wants for the narrow formats -- FP8 in memory, f32 in the accumulator.
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
    matmul_fp8!(a, b, c)

`c = a * b` with FP8 operands and an f32 accumulator.

FP8 is a storage format: there is no FP8 arithmetic to emit, so each element is
widened on load. Those conversions are single `arith.extf` ops rather than the
software conversion the FP8 package defines -- see `src/interpreter.jl`.
"""
function matmul_fp8!(
        a::Tile{Float8_E4M3FN, Tuple{M, K}}, b::Tile{Float8_E4M3FN, Tuple{K, N}},
        c::Tile{Float32, Tuple{M, N}},
    ) where {M, K, N}
    for i in 1:M, j in 1:N
        acc = zero(Float32)
        for k in 1:K
            acc += Float32(a[i, k]) * Float32(b[k, j])
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

if get(ENV, "IRON_RUN", "0") == "1"
    # Needs an NPU, XRT and the MLIR-AIE toolchain.
    T = Float32
    program = matmul_program(matmul!, square(T), square(T), square(T))
    compiled = IRON.compile(program)

    # Both operands are asymmetric and neither is the identity, so a transposed
    # tile or a swapped pair of operands changes the answer. Every value here is a
    # small integer and every partial sum stays well under 2^24, so the product is
    # exact in f32 and can be compared for equality. A symmetric `a` with `b = I`
    # is a trap: it is invariant under exactly the mistakes worth catching.
    a = T[T(10i + j) for i in 1:M, j in 1:K]
    b = T[T(i - 2j) for i in 1:K, j in 1:N]
    da, db = IRON.device_array(a), IRON.device_array(b)
    dc = IRON.device_zeros(square(T))
    IRON.run!(compiled, da, db, dc)

    result = IRON.host_array(dc)
    expected = a * b
    if result == expected
        println("NPU matmul matches")
    else
        wrong = count(!=(0), result .!= expected)
        println("MISMATCH in $wrong of $(length(expected)) elements")
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

    for T in (Int32, Float16, BFloat16, Float32)
        mlir = generate_mlir(matmul_program(matmul!, square(T), square(T), square(T)))
        println(rpad(string(T), 10), " -> ", join(mac(mlir), " ; "))
    end

    # FP8 operands, f32 accumulator.
    fp8 = generate_mlir(
        matmul_program(
            matmul_fp8!, square(Float8_E4M3FN), square(Float8_E4M3FN), square(Float32)
        )
    )
    println("\nFP8 kernel:")
    for line in split(fp8, '\n')
        occursin(r"arith\.(extf|mulf|addf)|memref\.load", line) && println("  ", strip(line))
    end
end
