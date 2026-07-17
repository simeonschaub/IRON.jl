# Executing a kernel, to check what it computes rather than what it looks like.
#
# The kernel is lowered to LLVM and JIT-compiled for the host. That is not the NPU,
# but it is the same scf/arith/memref the core is built from, so running it here
# validates the Julia -> MLIR translation on its own, without a device or the
# MLIR-AIE toolchain. A kernel that reads correctly and computes wrong -- an
# off-by-one in a subscript, a mixed-up loop-carried value -- fails here.

using IRON: Tile, compile_kernel!, context, memref_type, region
using MLIR: IR, API
using MLIR.Dialects: func

# The struct MLIR passes for a memref behind `llvm.emit_c_interface`: the pointers,
# the offset, then the sizes and strides. NTuple fields are laid out inline, so this
# matches the C struct.
struct MemRefDescriptor{T, N}
    allocated::Ptr{T}
    aligned::Ptr{T}
    offset::Int64
    sizes::NTuple{N, Int64}
    strides::NTuple{N, Int64}
end

# memrefs are row-major, Julia arrays are column-major, so `storage` holds the
# transposed buffer and the descriptor describes it with row-major strides.
function descriptor(storage::Array{T, N}, dims::NTuple{N, Int}) where {T, N}
    strides = ntuple(i -> prod(dims[(i + 1):end]; init = 1), N)
    ptr = pointer(storage)
    return MemRefDescriptor{T, N}(ptr, ptr, 0, Int64.(dims), Int64.(strides))
end

row_major(A::AbstractArray) = collect(permutedims(A, reverse(1:ndims(A))))
from_row_major(A::AbstractArray) = permutedims(A, reverse(1:ndims(A)))

"""
    jit_kernel(f, argtypes) -> Ptr

Compile kernel `f` for the host and return a pointer to its C interface. The
returned function takes one `MemRefDescriptor` pointer per argument.

The engine is kept alive by the returned closure's captured reference; dropping it
would free the code out from under the pointer.
"""
function jit_kernel(@nospecialize(f), @nospecialize(argtypes))
    API.mlirRegisterAllPasses()
    ctx = context()
    API.mlirRegisterAllLLVMTranslations(ctx)
    loc = IR.Location(; context = ctx)

    types = IR.Type[memref_type(ctx, T) for T in argtypes.parameters]
    block = IR.Block(types, [loc for _ in types])
    args = IR.Value[IR.argument(block, i) for i in eachindex(types)]
    compile_kernel!(ctx, block, f, argtypes, args)
    push!(block, func.return_(IR.Value[]; location = loc))

    fn = func.func_(;
        sym_name = IR.Attribute("kernel"; context = ctx),
        function_type = IR.FunctionType(types, IR.Type[]; context = ctx),
        body = region(block),
        location = loc,
    )
    # Discardable, not inherent: `llvm.emit_c_interface` belongs to the llvm dialect
    # rather than to func.func, and asks for the wrapper this test calls.
    IR.setattr!(fn, "llvm.emit_c_interface", IR.Attribute(API.mlirUnitAttrGet(ctx)))

    mod = IR.Module(loc)
    push!(IR.body(mod), fn)

    # The pipeline is added to the top-level manager unnested. Wrapping it in
    # `builtin.module(...)` would build a manager anchored at *contained* modules,
    # of which there are none, and every pass would silently not run.
    pm = IR.PassManager(; context = ctx)
    IR.add_pipeline!(
        IR.OpPassManager(pm),
        "convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm," *
            "finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts",
    )
    IR.run!(pm, mod)

    engine = IR.ExecutionEngine(mod, 2)
    fptr = IR.lookup(engine, "_mlir_ciface_kernel")
    fptr === nothing && error("IRON: could not find the compiled kernel")
    return engine, fptr
end

"""
    run_kernel(f, argtypes, arrays...) -> nothing

Compile and run `f` on the host over `arrays`, which are ordinary Julia arrays in
logical layout. Outputs are updated in place.
"""
function run_kernel(@nospecialize(f), @nospecialize(argtypes), arrays::AbstractArray...)
    engine, fptr = jit_kernel(f, argtypes)
    storage = [row_major(A) for A in arrays]
    descriptors = [
        Ref(descriptor(s, size(A))) for (s, A) in zip(storage, arrays)
    ]

    GC.@preserve engine storage descriptors begin
        ptrs = [Base.unsafe_convert(Ptr{Cvoid}, d) for d in descriptors]
        if length(ptrs) == 2
            @ccall $fptr(ptrs[1]::Ptr{Cvoid}, ptrs[2]::Ptr{Cvoid})::Cvoid
        elseif length(ptrs) == 3
            @ccall $fptr(
                ptrs[1]::Ptr{Cvoid}, ptrs[2]::Ptr{Cvoid}, ptrs[3]::Ptr{Cvoid}
            )::Cvoid
        else
            error("IRON: jit test supports 2 or 3 arguments, got $(length(ptrs))")
        end
    end

    for (A, s) in zip(arrays, storage)
        copyto!(A, from_row_major(s))
    end
    return nothing
end
