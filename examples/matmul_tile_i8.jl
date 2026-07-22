# The int8 counterpart of matmul_tile.jl: one AIE2P int8 matmul tile through `vmatmul`.
# AIE2P's int8 matrix-multiply is 8x8 * 8x8 -> 8x8 with an i32 accumulator (512 MACs/op,
# 4x the bf16 4x8x4 tile's 128), so the `Mat` shapes are 8x8 and the accumulator is Int32.
# `vmatmul` is element-type-generic -- it emits the same `vector.contract`, which
# `convert-vector-to-aievec` lowers to the i8 `aievec.matmul_aie2p` -- so this validates the
# int8 path end to end before wiring it into a tiled GEMM.
#
#   IRON_RUN=1 julia --project examples/matmul_tile_i8.jl

using IRON

# c := a * b for one int8 hardware matmul tile. `c` enters zeroed, so loading it as the
# accumulator gives a plain product; the accumulator is i32.
function mm_tile_i8!(
        a::Tile{Int8, Tuple{8, 8}}, b::Tile{Int8, Tuple{8, 8}},
        c::Tile{Int32, Tuple{8, 8}},
    )
    av = vload(Mat{8, 8, Int8}, a, 1, 1)
    bv = vload(Mat{8, 8, Int8}, b, 1, 1)
    acc = vload(Mat{8, 8, Int32}, c, 1, 1)
    vstore!(vmatmul(av, bv, acc), c, 1, 1)     # a * b
    return nothing
end

if get(ENV, "IRON_RUN", "0") == "1"
    # Small integers so the i32 accumulation of an 8-long dot product cannot overflow.
    a = Int8[(i + j) % 7 for i in 1:8, j in 1:8]
    b = Int8[(i - 2j) % 5 for i in 1:8, j in 1:8]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray(zeros(Int32, 8, 8))

    @iron flags = ["--alloc-scheme=basic-sequential"] mm_tile_i8!(In(da), In(db), Out(dc))

    result = Array(dc)
    expected = Int32.(a) * Int32.(b)
    if result == expected
        println("NPU int8 matmul tile (8x8 * 8x8 -> i32) matches")
    else
        println("MISMATCH")
        println("  got:      ", result)
        println("  expected: ", expected)
        result == permutedims(expected) && println("  (result is the transpose -> Mat load major order)")
    end
else
    println("Single AIE int8 matmul tile via `vmatmul`. Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/matmul_tile_i8.jl")
end
