# Probe the `dims_to_stream` DMA data-layout transform in isolation: copy a small buffer
# through a single core whose *input* object FIFO applies a `(size, stride)` access
# pattern, and print what comes out. This nails two things before wiring dims_to_stream
# into the GEMM: (1) that a non-empty `#aie<bd_dim_layout_array[...]>` compiles through
# aiecc, and (2) what the pattern actually does to the data.
#
#   IRON_RUN=1 julia --project examples/dims_probe.jl
#
# The pattern below is the whole_array A shape `[(m/r, r*k), (k/s, s), (r, k), (s, 1)]`
# scaled down to a 4x8 tile with r=4, s=8 -- i.e. a single 4x8 block, which should be an
# identity (a sanity check that the attribute compiles and does not corrupt data). Change
# `pattern` to explore other rearrangements.

using IRON

# Whole-buffer copy: `b := a`, element by element.
function copytile!(a::Tile{Int32, Tuple{R, C}}, b::Tile{Int32, Tuple{R, C}}) where {R, C}
    for i in 1:R, j in 1:C
        b[i, j] = a[i, j]
    end
    return nothing
end

if get(ENV, "IRON_RUN", "0") == "1"
    R, C = 4, 8
    a = Int32[10 * i + j for i in 1:R, j in 1:C]     # distinct values, easy to read
    da = NPUArray(a)
    db = NPUArray{Int32}(undef, Tile{Int32, Tuple{R, C}})

    # Access pattern applied to the input FIFO as the shim streams `a` to the core.
    # (size, stride) pairs, outermost first, in elements. Empty on the output.
    pattern = [(4, 8), (8, 1)]     # one 4x8 block, r=4 rows of s=8: contiguous -> identity

    @iron flags = ["--alloc-scheme=basic-sequential"] dims_to_stream = [pattern, Tuple{Int, Int}[]] copytile!(In(da), Out(db))

    got = Array(db)
    println("input a =")
    display(a); println()
    println("output b (a streamed through dims_to_stream $pattern) =")
    display(got); println()
    println(got == a ? "identity (b == a)" : "rearranged (b != a)")
else
    println("dims_to_stream probe. Run on an NPU with:")
    println("  IRON_RUN=1 julia --project examples/dims_probe.jl")
end
