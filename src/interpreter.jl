# A custom abstract interpreter, so that kernels can be inferred against device
# semantics rather than host ones.
#
# The motivating case is the FP8 formats. Their Julia packages implement
# arithmetic and conversion in software -- unpacking to bits, renormalizing,
# branching -- because a CPU has no FP8. Inferring a kernel against those methods
# buries a single hardware conversion under a few hundred integer ops. An overlay
# method table replaces just those methods with intrinsics the kernel compiler
# knows how to emit, and leaves everything else alone.
#
# Nothing is needed for the standard formats: Julia lowers `Float16`, `Float32`,
# `Float64` and `BFloat16` arithmetic to intrinsics already, since LLVM has all
# four natively.

"""
    IRONMethodTable

Methods overlaid onto the kernel's view of Base. Add to it with
`Base.Experimental.@overlay IRONMethodTable ...`.
"""
Base.Experimental.@MethodTable IRONMethodTable

"""
    IRONInterpreter(; world)

An `AbstractInterpreter` that applies [`IRONMethodTable`](@ref). A custom type is
needed because `NativeInterpreter` fixes its method table.
"""
struct IRONInterpreter <: CC.AbstractInterpreter
    world::UInt
    method_table::CC.CachedMethodTable{CC.OverlayMethodTable}
    inf_cache::@static isdefined(CC, :InferenceCache) ? CC.InferenceCache :
        Vector{CC.InferenceResult}
    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

function IRONInterpreter(; world::UInt = Base.get_world_counter())
    table = CC.CachedMethodTable(CC.OverlayMethodTable(world, IRONMethodTable))
    inf_cache = @static if isdefined(CC, :InferenceCache)
        CC.InferenceCache()
    else
        Vector{CC.InferenceResult}()
    end
    return IRONInterpreter(
        world, table, inf_cache, CC.InferenceParams(), CC.OptimizationParams()
    )
end

CC.InferenceParams(interp::IRONInterpreter) = interp.inf_params
CC.OptimizationParams(interp::IRONInterpreter) = interp.opt_params
CC.get_inference_cache(interp::IRONInterpreter) = interp.inf_cache
CC.method_table(interp::IRONInterpreter) = interp.method_table
CC.cache_owner(::IRONInterpreter) = IRONCacheToken()

@static if isdefined(CC, :get_inference_world)
    CC.get_inference_world(interp::IRONInterpreter) = interp.world
else
    CC.get_world_counter(interp::IRONInterpreter) = interp.world
end

# Keeps inference results for kernels out of the native cache, which holds results
# inferred without the overlay.
struct IRONCacheToken end

@static if !isdefined(CC, :InferenceCache)
    CC.lock_mi_inference(::IRONInterpreter, ::Core.MethodInstance) = nothing
    CC.unlock_mi_inference(::IRONInterpreter, ::Core.MethodInstance) = nothing
end

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
