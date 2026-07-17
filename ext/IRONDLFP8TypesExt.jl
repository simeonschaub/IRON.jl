module IRONDLFP8TypesExt

import IRON

using MLIR: IR
using DLFP8Types: Float8_E4M3FN, Float8_E5M2

# DLFP8Types spells the two formats MLIR has as Float8_E4M3FN and Float8_E5M2. Its
# FNUZ variants have no MLIR 18 equivalent, so they fall through to the
# unsupported-type error rather than being mapped onto a neighbouring format.
const FP8_TYPES = (Float8_E4M3FN, Float8_E5M2)

IRON.mlir_eltype(ctx::IR.Context, ::Type{Float8_E4M3FN}) = IR.Float8E4M3FN(; context = ctx)
IRON.mlir_eltype(ctx::IR.Context, ::Type{Float8_E5M2}) = IR.Float8E5M2(; context = ctx)

# Replace the package's software conversions with the hardware one, so that a
# kernel converting FP8 to f32 emits a single arith.extf rather than a few hundred
# integer ops. The `@eval` has to happen here rather than in a helper inside IRON,
# which is closed for evaluation while this extension precompiles.
for F8 in FP8_TYPES, F in IRON.STANDARD_FLOATS
    @eval Base.Experimental.@overlay IRON.IRONMethodTable $F(x::$F8) = IRON.convert_float(x, $F)
    @eval Base.Experimental.@overlay IRON.IRONMethodTable $F8(x::$F) = IRON.convert_float(x, $F8)
end

end
