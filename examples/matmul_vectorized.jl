# A vectorized matrix multiply, reaching the AIE vector unit from Julia.
#
# The scalar unit on an AIE2 core cannot multiply floats: examples/diagnose.jl
# shows an f32 matmul returning wrong data while the same kernel over integers is
# correct. Float throughput lives in the vector unit, which the C++ kernels reach
# through `aie::mmul`. This gets there without emitting a single `aievec` op,
# because aiecc already runs `convert-vector-to-aievec` over every AIE2/AIE2p core
# and that pipeline "ingests arbitrary MLIR Vector code":
#
#     arith.extf + vector.fma  ->  aievec.mac_elem  ->  the MAC intrinsic
#
# Two rules shape the kernel below, both from
# lib/Dialect/AIEVec/Transforms/VectorToAIEVecConversions.cpp:
#
#  1. The vector width is the hardware's, not the algorithm's. vector.fma lowers
#     only for f32 at 16 lanes, and bf16 at 16 or 32 -- AIE2's vector registers are
#     512 bits. A vector<8xf32> matches no pattern and aiecc stops with "failed to
#     legalize operation 'vector.fma'".
#
#  2. An f32 vector.fma lowers only when *both* operands come from an arith.extf on
#     bf16, and says so: "vector.fma operands are f32, and they don't come from
#     arith.extf on bf16; can't lower to aievec." There is no f32 multiplier
#     anywhere on the core -- the MAC multiplies bf16 and accumulates into f32,
#     which is the same fact as `_MM_COMBOS` listing (bfloat16, float32) and no f32
#     entry.
#
# So the operands are bf16, widened per vector, and the accumulator is f32: the
# mixed precision the hardware is built around. `Vec{16,BFloat16}` is how a kernel
# says it -- IRON's own Vec rather than SIMD.jl's, whose element type must come
# from a fixed list that BFloat16 is not on.
#
# Compiling and running need the AIE toolchain JLLs and an NPU -- but no Python:
#   julia --project examples/matmul_vectorized.jl

using IRON
using BFloat16s: BFloat16
using DLFP8Types: Float8_E4M3FN, Float8_E5M2

"""
    tile_size(Tacc) -> Int

How wide a tile is, which is how wide a vector is: one 512-bit AIE2 register holds
16 f32 or i32, 32 bf16 or i16, 64 i8. The accumulator sets it, since that is what
the kernel carries across the k loop.
"""
tile_size(::Type{Tacc}) where {Tacc} = 512 ÷ (8 * sizeof(Tacc))

"""
    mac_via(T, Tacc) -> Type

The type an operand of type `T` is widened through on its way to accumulator
`Tacc`.

The f32 `vector.fma` pattern does not ask merely for widened operands, but for
operands widened *from bf16* -- that being the only thing the MAC multiplies. So
FP8 widens twice, `f8 -> bf16 -> f32`. Everything else goes straight to the
accumulator, and the extra step costs nothing, because `Vec{N,T}(::Vec{N,T})`
emits no op at all.
"""
mac_via(::Type{T}, ::Type{Tacc}) where {T, Tacc} = Tacc
mac_via(::Type{Float8_E4M3FN}, ::Type{Float32}) = BFloat16
mac_via(::Type{Float8_E5M2}, ::Type{Float32}) = BFloat16

"""
    matmul_vec!(a, b, c)

`c = a * b` with operands of type `T` and an accumulator of type `Tacc`, a column of
`c` at a time. One kernel for i16 -> i32, bf16 -> f32 and FP8 -> f32.

A tile is column-major, so the contiguous vector is a *column*: `vload` reads down a
column of `a`, `b[k, j]` is broadcast, and the accumulator carries a whole column of
`c`. Passing column-major Julia arrays (the norm) and reading the result back
column-major then needs no transpose and no operand swap -- `c` really is `a * b`.

Widen the vectors, never the scalar. Both MAC patterns -- `vector.fma` for floats
and `arith.muli`+`arith.addi` for integers -- ask the same thing of their operands:

    "widening ops in the `lhs` and `rhs` operands, or fail otherwise"

so each must be an `arith.extf`/`extsi` over a vector. Broadcasting `b[k, j]`
straight into `Tacc` widens the scalar and then splats it, which leaves the
multiply in the accumulator's type, matching nothing:

    %17 = arith.extsi %16 : i16 to i32              # scalar
    %18 = vector.broadcast %17 : i32 to vector<16xi32>
    %20 = arith.muli %18, %19 : vector<16xi32>      # no i32 multiplier exists

and peano stops with `unable to legalize <16 x s32> G_MUL` -- the integer echo of
the f32 story. Broadcasting in `T` first and widening the vector afterwards puts
the multiply back where the hardware has one.
"""
function matmul_vec!(
        a::Tile{T, Tuple{M, K}}, b::Tile{T, Tuple{K, N}}, c::Tile{Tacc, Tuple{M, N}}
    ) where {T, Tacc, M, K, N}
    Mid = mac_via(T, Tacc)
    for j in 1:N
        acc = zero(Vec{M, Tacc})
        for k in 1:K
            av = vload(Vec{M, T}, a, 1, k)          # vector.load, a column of `a`, in T
            bv = Vec{M, T}(b[k, j])                 # vector.broadcast, in T
            acc = muladd(                           # widen the vectors, then MAC
                Vec{M, Tacc}(Vec{M, Mid}(av)),
                Vec{M, Tacc}(Vec{M, Mid}(bv)),
                acc,
            )
        end
        vstore!(acc, c, 1, j)
    end
    return nothing
end

function matmul_program(kernel, ::Type{Tin}, ::Type{Tacc}) where {Tin, Tacc}
    S = tile_size(Tacc)
    A = Tile{Tin, Tuple{S, S}}
    C = Tile{Tacc, Tuple{S, S}}
    of_a, of_b, of_c = ObjectFifo{A}("a"), ObjectFifo{A}("b"), ObjectFifo{C}("c")

    rt = Runtime()
    start!(rt, Worker(kernel, [consumer(of_a), consumer(of_b), producer(of_c)]))
    fill!(rt, producer(of_a), 1)
    fill!(rt, producer(of_b), 2)
    drain!(rt, consumer(of_c), 3)

    return Program(npu2, rt, [A, A, C])
end

# Bank-aware allocation, the default, silently overlaps the object FIFO buffers for
# a design with three FIFOs on one core; every matrix multiply under
# programming_examples/ passes this flag.
const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

# Every value here is a small integer, exact in bf16 and in FP8's 4-bit mantissa,
# so each product can be compared for equality rather than tolerance.
operand(::Type{T}, ::Type{Tacc}, f) where {T, Tacc} =
    T[T(f(i, j)) for i in 1:tile_size(Tacc), j in 1:tile_size(Tacc)]

function run_case(::Type{Tin}, ::Type{Tacc}) where {Tin, Tacc}
    label = rpad("$Tin -> $Tacc", 26)
    a = operand(Tin, Tacc, (i, j) -> (i + j) % 7)
    b = operand(Tin, Tacc, (i, j) -> (i - 2j) % 5)

    compiled = IRON.compile(
        matmul_program(matmul_vec!, Tin, Tacc); flags = AIECC_FLAGS,
    )
    dc = NPUArray{Tacc}(undef, Tile{Tacc, Tuple{tile_size(Tacc), tile_size(Tacc)}})
    IRON.run!(compiled, NPUArray(a), NPUArray(b), dc)

    result = Array(dc)
    expected = Tacc.(Float32.(a) * Float32.(b))
    if result == expected
        println(label, "PASS")
    else
        println(label, "MISMATCH in $(count(result .!= expected)) of $(length(expected))")
        println("  got:      ", result)
        println("  expected: ", expected)
    end
    return nothing
end

if get(ENV, "IRON_RUN", "0") == "1"
    # Measured on an NPU2. Only bf16 -> f32 makes it all the way, which is the
    # combination the MAC is built out of; the other two are here because where
    # they stop is informative, and they throw rather than being caught, since the
    # backtrace is the useful part.
    #
    #   BFloat16 -> Float32       PASS
    #   Int16 -> Int16            same type, so nothing widens and the multiply stays
    #                             in i16 -- the type the vector unit multiplies. This
    #                             is the integer workaround; it wraps rather than
    #                             widening, so the products have to fit in i16.
    #   Int16 -> Int32            peano: "unable to legalize <16 x s32> G_MUL".
    #                             Nothing converts an integer multiply on AIE2p
    #                             (configureAIEVecV2PLegalizations has no MulIOp
    #                             rule), so it reaches peano as a plain LLVM mul and
    #                             there is no i32 vector multiply to select.
    #   Float8_E4M3FN -> Float32  the f8 -> bf16 widening becomes an aievec.ups whose
    #                             result type is not widened at all, because
    #                             getVectorOpDestType only widens 16-bit floats.
    #
    run_case(BFloat16, Float32)
    #run_case(Float8_E4M3FN, Float32)
    #run_case(Int16, Int16)
    #run_case(Int16, Int32)
else
    show_ops(mlir) = for l in split(mlir, '\n')
        occursin(r"vector\.(load|store|broadcast|fma)|arith\.(extf|extsi|muli|addi) .*vector", l) &&
            println("    ", strip(replace(l, r"^\s*%\w+ = " => "")))
    end

    for (Tin, Tacc) in (
            (Int16, Int16), (Int16, Int32), (BFloat16, Float32), (Float8_E4M3FN, Float32),
        )
        println("  $Tin -> $Tacc:")
        println(generate_mlir(matmul_program(matmul_vec!, Tin, Tacc)))
        println()
    end
end
