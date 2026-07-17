# Vector kernels.
#
# The scalar unit on an AIE2 core cannot multiply floats -- an f32 matmul written
# with scalar arithmetic compiles, runs, and returns wrong data, while the same
# kernel over integers is correct. Float throughput lives in the vector unit, and
# reaching it needs no knowledge of `aievec` here: aiecc already runs
# `convert-vector-to-aievec` over every AIE2/AIE2p core, and that pipeline "ingests
# arbitrary MLIR Vector code". So a kernel that says `vector<16xbf16>` arrives at
# the same `mac_elem` intrinsic the C++ kernels reach through `aie::mmul`.
#
# `Vec{N,T}` is how a kernel says that. It is IRON's own rather than `SIMD.Vec`,
# which restricts its element type to a fixed list that `BFloat16` is not on --
# and bf16 is not incidental here but the whole point, since the MAC multiplies
# bf16 and accumulates into f32, and an f32 `vector.fma` lowers *only* when both
# operands come from an `arith.extf` on bf16.
#
# Like `Tile`, this is a marker type: it is never constructed and never runs. Its
# operators exist to be inferred, and the kernel compiler rewrites them into the
# `vector` dialect.

"""
    Vec{N,T}

A vector of `N` lanes of `T`, as seen from inside a kernel: `vector<NxT>`.

`N` is the *hardware's* width, not the algorithm's. `convert-vector-to-aievec`
lowers `vector.fma` only for f32 at 16 lanes and bf16 at 16 or 32, matching AIE2's
512-bit vector registers; a `vector<8xf32>` matches no pattern and aiecc stops with
`failed to legalize operation 'vector.fma'`.

Any element type [`mlir_eltype`](@ref) knows is allowed, `BFloat16` and the FP8
formats included.
"""
struct Vec{N, T}
    # Never read, and no `Vec` is ever built. The field is here so that `Vec` is not
    # a singleton: a type with one inhabitant lets inference replace any value of it
    # with that constant, and every intrinsic below would come back as a folded
    # `Vec{N,T}()` rather than a value the kernel compiler can lower. `Tile` gets
    # away with being empty only because nothing returns one.
    data::NTuple{N, T}
end

Base.eltype(::Type{Vec{N, T}}) where {N, T} = T
Base.length(::Type{Vec{N, T}}) where {N, T} = N
lanes(::Type{Vec{N, T}}) where {N, T} = N

# The intrinsics. Each is `@noinline` so the call survives into the IR, and returns
# through `inferencebarrier` so inference cannot fold it away. None may construct a
# `Vec`: `zero(Vec{N,T})` is defined below in terms of `vbroadcast`, so building one
# here would recur, and inference would conclude the intrinsic always throws -- which
# surfaces as a kernel that "must return nothing, got Union{}". They launder an
# argument instead: the typeassert names the result type without making a value.

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

Splat scalar `x` across `N` lanes, as one `vector.broadcast`.
"""
@noinline function vbroadcast(::Type{Vec{N, T}}, x) where {N, T}
    return Base.inferencebarrier(x)::Vec{N, T}
end

"""
    vconvert(Vec{N,T}, v) -> Vec{N,T}

Convert `v` lane-wise to element type `T`, as one `arith.extf`/`truncf` over
vectors.

Widening bf16 to f32 is the case that matters: it is what makes an f32
`vector.fma` legal, and the only way to a float multiply on this hardware.
"""
@noinline function vconvert(::Type{Vec{N, T}}, v::Vec{N, S}) where {N, T, S}
    return Base.inferencebarrier(v)::Vec{N, T}
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

`a * b + c` as one `vector.fma`, the multiply-accumulate the vector unit is built
around.
"""
@noinline function vfma(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N, T}
    return Base.inferencebarrier(a)::Vec{N, T}
end

# The surface a kernel writes. These inline away, leaving the intrinsics; because
# `Vec` is IRON's own type, they are ordinary methods rather than overlays.
Base.:+(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vadd(a, b)
Base.:-(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vsub(a, b)
Base.:*(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vmul(a, b)
Base.:/(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vdiv(a, b)
Base.muladd(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N, T} = vfma(a, b, c)
Base.fma(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N, T} = vfma(a, b, c)
Base.sum(v::Vec{N, T}) where {N, T} = vreduce_add(v)
Base.zero(::Type{Vec{N, T}}) where {N, T} = vbroadcast(Vec{N, T}, zero(T))
Base.one(::Type{Vec{N, T}}) where {N, T} = vbroadcast(Vec{N, T}, one(T))

"""
    Vec{N,T}(x::Number) -> Vec{N,T}

Splat `x` across `N` lanes.
"""
Vec{N, T}(x::Number) where {N, T} = vbroadcast(Vec{N, T}, convert(T, x))

"""
    Vec{N,T}(v::Vec{N,S}) -> Vec{N,T}

Convert `v` lane-wise to `T`.
"""
Vec{N, T}(v::Vec{N, S}) where {N, T, S} = vconvert(Vec{N, T}, v)
