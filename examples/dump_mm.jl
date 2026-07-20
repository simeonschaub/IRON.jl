# Print the MLIR IRON emits for the single-tile matmul kernel, without invoking aiecc --
# a diagnostic for the `vmatmul` lowering.  Run:
#   julia --project=examples examples/dump_mm.jl

using IRON
using IRON: context, memref_type, compile_kernel!, region
using MLIR: IR
using MLIR.Dialects: func
using BFloat16s: BFloat16

function mm_tile!(
        a::Tile{BFloat16, Tuple{4, 8}}, b::Tile{BFloat16, Tuple{8, 4}},
        c::Tile{Float32, Tuple{4, 4}},
    )
    # `Mat{R,C}` is a Julia RxC matrix; `vmatmul` hides the column-major transpose.
    av = vload(Mat{4, 8, BFloat16}, a, 1, 1)
    bv = vload(Mat{8, 4, BFloat16}, b, 1, 1)
    acc = vload(Mat{4, 4, Float32}, c, 1, 1)
    vstore!(vmatmul(av, bv, acc), c, 1, 1)     # a * b
    return nothing
end

argtypes = Tuple{
    Tile{BFloat16, Tuple{4, 8}}, Tile{BFloat16, Tuple{8, 4}}, Tile{Float32, Tuple{4, 4}},
}

ctx = context()
loc = IR.Location(; context = ctx)
types = IR.Type[memref_type(ctx, T) for T in argtypes.parameters]
block = IR.Block(types, [loc for _ in types])
args = IR.Value[IR.argument(block, i) for i in eachindex(types)]

compile_kernel!(ctx, block, mm_tile!, argtypes, args)
push!(block, func.return_(IR.Value[]; location = loc))

fn = func.func_(;
    sym_name = IR.Attribute("kernel"; context = ctx),
    function_type = IR.FunctionType(types, IR.Type[]; context = ctx),
    body = region(block),
    location = loc,
)
mod = IR.Module(loc)
push!(IR.body(mod), fn)
print(string(IR.Operation(mod)))
