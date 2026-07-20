# The smallest use of the AIE matrix-multiply unit: one 4x8 * 8x4 -> 4x4 tile through
# `vmatmul`, which lowers to `aievec.matmul_aie2p`. This validates the intrinsic end to
# end -- does the hardware matmul compute `A * B` -- before wiring it into a tiled GEMM.
#
#   IRON_RUN=1 julia --project examples/matmul_tile.jl
#
# If the result comes back transposed, the `Mat` load is reading the tile in the wrong
# major order and the fix is a `dims_to_stream` (or a transposed load) -- that layout
# plumbing is the next increment; here we just check the op itself.

using IRON
using BFloat16s: BFloat16

# c := a * b for one hardware matmul tile. `c` enters zeroed, so loading it as the
# accumulator gives a plain product.
function mm_tile!(
        a::Tile{BFloat16, Tuple{4, 8}}, b::Tile{BFloat16, Tuple{8, 4}},
        c::Tile{Float32, Tuple{4, 4}},
    )
    # Tiles are column-major, so a `Tile{R,C}` presents to the matmul unit as its `C`x`R`
    # transpose. Load each operand at that register shape and multiply `bᵀ · aᵀ`, which is
    # `(a·b)ᵀ` -- exactly the transpose `c`'s column-major memref already expects, so it
    # reads back as `a·b`.
    av = vload(Mat{8, 4, BFloat16}, a, 1, 1)   # aᵀ
    bv = vload(Mat{4, 8, BFloat16}, b, 1, 1)   # bᵀ
    acc = vload(Mat{4, 4, Float32}, c, 1, 1)
    vstore!(vmatmul(bv, av, acc), c, 1, 1)     # bᵀ·aᵀ = (a·b)ᵀ
    return nothing
end

if get(ENV, "IRON_RUN", "0") == "1"
    # Small integers, exact in bf16, so the product can be compared for equality.
    a = BFloat16[(i + j) % 7 for i in 1:4, j in 1:8]
    b = BFloat16[(i - 2j) % 5 for i in 1:8, j in 1:4]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray(zeros(Float32, 4, 4))

    @iron flags = ["--alloc-scheme=basic-sequential"] mm_tile!(In(da), In(db), Out(dc))

    result = Array(dc)
    expected = Float32.(a) * Float32.(b)
    if result == expected
        println("NPU matmul tile (4x8 * 8x4) matches")
    else
        println("MISMATCH")
        println("  got:      ", result)
        println("  expected: ", expected)
        # A transpose of `expected` matching `result` would point at the load major order.
        result == permutedims(expected) && println("  (result is the transpose -> Mat load major order)")
    end
else
    println("Single AIE matmul tile via `vmatmul`. Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/matmul_tile.jl")
end
