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
