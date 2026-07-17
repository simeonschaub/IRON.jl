module IRONMicrofloatsExt

import IRON
import Microfloats

using MLIR: IR
using Microfloats: Float8_E4M3FN, Float8_E5M2

# Only the two formats MLIR 18 has are mapped. Microfloats' other variants
# (Float8_E4M3 IEEE, Float8_E3M4, Float6_*, Float4_*, Float8_E8M0FNU) have no
# equivalent there, so they fall through to the unsupported-type error rather than
# being mapped onto a neighbouring format.
const FP8_TYPES = (Float8_E4M3FN, Float8_E5M2)

IRON.mlir_eltype(ctx::IR.Context, ::Type{Float8_E4M3FN}) = IR.Float8E4M3FN(; context = ctx)
IRON.mlir_eltype(ctx::IR.Context, ::Type{Float8_E5M2}) = IR.Float8E5M2(; context = ctx)

# numpy has no FP8; ml_dtypes supplies the same two formats under these names.
IRON.numpy_dtype(::Type{Float8_E4M3FN}) = IRON.ml_dtypes().float8_e4m3fn
IRON.numpy_dtype(::Type{Float8_E5M2}) = IRON.ml_dtypes().float8_e5m2
IRON.host_values(A::AbstractArray{<:Union{FP8_TYPES...}}) = Float32.(A)

# Microfloats are stored one per byte, so the default `8 * sizeof` over-counts the
# sub-byte formats. The width is what decides widening against narrowing.
IRON.bitwidth(::Type{T}) where {T <: Microfloats.Microfloat} = Microfloats.bitwidth(T)

# See ext/IRONDLFP8TypesExt.jl: the `@eval` belongs here, not in a helper inside
# IRON, which is closed for evaluation while this extension precompiles.
for F8 in FP8_TYPES, F in IRON.STANDARD_FLOATS
    @eval Base.Experimental.@overlay IRON.IRONMethodTable $F(x::$F8) = IRON.convert_float(x, $F)
    @eval Base.Experimental.@overlay IRON.IRONMethodTable $F8(x::$F) = IRON.convert_float(x, $F8)
end

end
