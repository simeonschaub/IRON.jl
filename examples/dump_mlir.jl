# Write out the MLIR for each vectorized matmul, to run aie-opt against by hand.
#
# Generating MLIR needs no NPU and no toolchain, so this runs anywhere. Compare the
# three: bf16 -> f32 reaches the vector MAC, and the other two do not.
#
#   julia --project=examples examples/dump_mlir.jl /tmp
#
# Then, for each file (see the README section "Vector kernels"):
#
#   aie-opt --convert-vector-to-aievec='aie-target=aie2p' /tmp/mm_i16.mlir

using IRON
using BFloat16s: BFloat16
using DLFP8Types: Float8_E4M3FN

const M, K, N = 16, 16, 16

mac_via(::Type{T}, ::Type{Tacc}) where {T, Tacc} = Tacc
mac_via(::Type{Float8_E4M3FN}, ::Type{Float32}) = BFloat16

function matmul_vec!(
        a::Tile{T, Tuple{M, K}}, b::Tile{T, Tuple{K, N}}, c::Tile{Tacc, Tuple{M, N}}
    ) where {T, Tacc, M, K, N}
    Mid = mac_via(T, Tacc)
    for i in 1:M
        acc = zero(Vec{N, Tacc})
        for k in 1:K
            av = Vec{N, T}(a[i, k])
            bv = vload(Vec{N, T}, b, k, 1)
            acc = muladd(Vec{N, Tacc}(Vec{N, Mid}(av)), Vec{N, Tacc}(Vec{N, Mid}(bv)), acc)
        end
        vstore!(acc, c, i, 1)
    end
    return nothing
end

# The scalar kernel, for the f32 case that runs and returns wrong data.
function matmul_scalar!(
        a::Tile{T, Tuple{M, K}}, b::Tile{T, Tuple{K, N}}, c::Tile{T, Tuple{M, N}}
    ) where {T, M, K, N}
    for i in 1:M, j in 1:N
        acc = zero(T)
        for k in 1:K
            acc += a[i, k] * b[k, j]
        end
        c[i, j] = acc
    end
    return nothing
end

function matmul_program(kernel, ::Type{Tin}, ::Type{Tacc}) where {Tin, Tacc}
    A = Tile{Tin, Tuple{M, K}}
    C = Tile{Tacc, Tuple{M, N}}
    of_a, of_b, of_c = ObjectFifo{A}("a"), ObjectFifo{A}("b"), ObjectFifo{C}("c")

    rt = Runtime()
    start!(rt, Worker(kernel, [consumer(of_a), consumer(of_b), producer(of_c)]))
    fill!(rt, producer(of_a), 1)
    fill!(rt, producer(of_b), 2)
    drain!(rt, consumer(of_c), 3)

    return Program(npu2, rt, [A, A, C])
end

outdir = get(ARGS, 1, ".")
cases = [
    ("mm_i16", matmul_vec!, Int16, Int32),
    ("mm_bf16", matmul_vec!, BFloat16, Float32),
    ("mm_fp8", matmul_vec!, Float8_E4M3FN, Float32),
    ("mm_f32_scalar", matmul_scalar!, Float32, Float32),
]

for (name, kernel, Tin, Tacc) in cases
    path = joinpath(outdir, "$name.mlir")
    write(path, generate_mlir(matmul_program(kernel, Tin, Tacc)))
    println(rpad("$Tin -> $Tacc", 26), path)
end
