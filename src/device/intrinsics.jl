"""
    convert_float(x, T) -> T

Convert `x` to floating-point type `T` in one hardware operation, lowering to
`arith.extf` or `arith.truncf`.

This is the intrinsic the FP8 conversion overlays route to, in place of the
software conversion their packages provide. It is only meaningful inside a
kernel: like tile indexing, the body exists to be inferred, not to run.
"""
@noinline function convert_float(x, ::Type{T}) where {T}
    return Base.inferencebarrier(zero(T))::T
end

"""
    STANDARD_FLOATS

The float types Julia and LLVM handle natively, and so the ones an FP8 value is
converted to in order to be computed with.
"""
const STANDARD_FLOATS = (Float16, Core.BFloat16, Float32, Float64)

# An extension adding an FP8 format routes both directions of conversion against
# STANDARD_FLOATS to `convert_float`, replacing the software conversions its
# package defines. The `@eval` loop has to sit in the extension's own module: a
# helper here would evaluate into IRON, which is closed during the extension's
# precompilation. See ext/ for the pattern.
#
# FP8 arithmetic itself is deliberately left alone: the hardware computes in f16 or
# f32 and treats FP8 as a storage format, which is the same split cuTile makes, so
# a kernel converts on load and accumulates in a wider type.
