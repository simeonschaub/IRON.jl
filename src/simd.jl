# Vector kernels.
#
# The scalar unit on an AIE2 core cannot multiply floats -- an f32 or bf16 matmul
# written with scalar arithmetic compiles, runs, and returns wrong data, while the
# same kernel over integers is correct. Floating-point throughput lives in the
# vector unit, and the way to reach it is the `vector` dialect: aiecc runs
# `convert-vector-to-aievec` over every AIE2/AIE2p core, and that pipeline
# "ingests arbitrary MLIR Vector code" and lowers it to `aievec` ops -- the same
# `mul_elem`/`mac_elem` intrinsics the hand-written C++ kernels use through
# `aie::mmul`. So a kernel that says `vector<8xf32>` gets there without this
# package knowing anything about `aievec`.
#
# `SIMD.Vec{N,T}` is the Julia side of that. Its arithmetic normally inlines
# straight to `llvmcall` with a literal LLVM IR string, which carries no meaning
# here, so the overlay method table redirects it to the intrinsics below before
# inlining can happen -- the same trick the FP8 conversions use.

using SIMD: Vec

"""
    lanes(::Type{Vec{N,T}}) -> Int

The width of a vector type.
"""
lanes(::Type{Vec{N, T}}) where {N, T} = N

# The intrinsics. Like tile indexing, these exist to be inferred rather than run:
# each is `@noinline` so the call survives into the IR, and returns through
# `inferencebarrier` so inference cannot fold it away.
#
# None of these bodies may construct a `Vec`. `zero(Vec{N,T})` is
# `Vec{N,T}(zero(T))`, which the overlay below sends straight back here, and the
# recursion makes inference conclude the intrinsic always throws -- which reads as
# a kernel that "must return nothing, got Union{}". They launder an argument
# through `inferencebarrier` instead: the typeassert names the result type without
# building a value.

"""
    vload(Vec{N,T}, tile, I...) -> Vec{N,T}

Read `N` contiguous elements from `tile` starting at `I`, as one `vector.load`.
"""
@noinline function vload(::Type{Vec{N, T}}, tile::Tile, I::Int...) where {N, T}
    return Base.inferencebarrier(tile)::Vec{N, T}
end

"""
    vstore!(v, tile, I...)

Write the lanes of `v` to `tile` starting at `I`, as one `vector.store`.
"""
@noinline function vstore!(v::Vec{N, T}, tile::Tile, I::Int...) where {N, T}
    Base.donotdelete(v, tile, I)
    return nothing
end

"""
    vbroadcast(Vec{N,T}, x) -> Vec{N,T}

Splat scalar `x` across `N` lanes.
"""
@noinline function vbroadcast(::Type{Vec{N, T}}, x) where {N, T}
    return Base.inferencebarrier(x)::Vec{N, T}
end

"""
    vreduce_add(v) -> T

Sum the lanes of `v`, as one `vector.reduction`.
"""
@noinline function vreduce_add(v::Vec{N, T}) where {N, T}
    return Base.inferencebarrier(zero(T))::T
end

for op in (:vadd, :vsub, :vmul, :vdiv)
    @eval @noinline function $op(a::Vec{N, T}, b::Vec{N, T}) where {N, T}
        return Base.inferencebarrier(a)::Vec{N, T}
    end
end

"""
    vfma(a, b, c) -> Vec

`a * b + c` as one `vector.fma`, which is what the multiply-accumulate the vector
unit is built around lowers from.
"""
@noinline function vfma(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N, T}
    return Base.inferencebarrier(a)::Vec{N, T}
end

# Redirect SIMD.jl's arithmetic onto the intrinsics. Without this the operators
# inline to `llvmcall`, which the kernel compiler cannot read.
Base.Experimental.@overlay IRONMethodTable Base.:+(a::Vec{N, T}, b::Vec{N, T}) where {N, T} =
    vadd(a, b)
Base.Experimental.@overlay IRONMethodTable Base.:-(a::Vec{N, T}, b::Vec{N, T}) where {N, T} =
    vsub(a, b)
Base.Experimental.@overlay IRONMethodTable Base.:*(a::Vec{N, T}, b::Vec{N, T}) where {N, T} =
    vmul(a, b)
Base.Experimental.@overlay IRONMethodTable Base.:/(a::Vec{N, T}, b::Vec{N, T}) where {N, T} =
    vdiv(a, b)
Base.Experimental.@overlay IRONMethodTable Base.muladd(
    a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}
) where {N, T} = vfma(a, b, c)
Base.Experimental.@overlay IRONMethodTable Base.fma(
    a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}
) where {N, T} = vfma(a, b, c)
Base.Experimental.@overlay IRONMethodTable Base.sum(v::Vec{N, T}) where {N, T} =
    vreduce_add(v)
Base.Experimental.@overlay IRONMethodTable Vec{N, T}(x::Number) where {N, T} =
    vbroadcast(Vec{N, T}, T(x))
