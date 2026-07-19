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

"""
    Mat{R, C, T}

An `R`x`C` matrix tile held in a hardware vector register -- a rank-2 `vector<RxCxT>`.
Unlike [`Vec`](@ref) (a 1-D SIMD vector), a `Mat` is the shape the AIE matrix-multiply
unit consumes: [`vmatmul`](@ref) takes an `R`x`K` and a `K`x`C` `Mat` and accumulates
into an `R`x`C` one. The valid shapes/types are the hardware's -- for AIE2P bf16 that is
`4`x`8` * `8`x`4` -> `4`x`4` f32.
"""
struct Mat{R, C, T}
    # Non-singleton for the same reason as `Vec` (see above); never read.
    data::NTuple{R, NTuple{C, T}}
end

Base.eltype(::Type{Mat{R, C, T}}) where {R, C, T} = T
Base.size(::Type{Mat{R, C, T}}) where {R, C, T} = (R, C)

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
    vreinterpret(Vec{N,S}, v::Vec{N,T}) -> Vec{N,S}

Reinterpret the bits of `v` as element type `S` (which must be the same width), as
one `arith.bitcast` over vectors -- no numeric conversion, just a relabel.

This is the door to bit-level tricks the hardware needs. An f32 vector max, for one,
does not lower (`aievec.max` has no f32 form), but reading the bits as i32 -- where a
negative float is a negative integer -- turns `relu` into an i32 `max`, which does.
"""
@noinline function vreinterpret(::Type{Vec{N, S}}, v::Vec{N, T}) where {N, S, T}
    return Base.inferencebarrier(v)::Vec{N, S}
end

"""
    vexp(v::Vec{N,T}) -> Vec{N,T}

Elementwise `exp`, as one `math.exp` over the vector.

The vector unit has a real exp: `convert-vector-to-aievec` lowers `math.exp` to the
hardware exp on AIE2/AIE2p -- but **only for bf16** at 16 or 32 lanes (an f32 or
scalar `math.exp` has no such lowering and reaches no exp instruction). So `exp` on
this hardware means `exp` of a `Vec{16,BFloat16}`/`Vec{32,BFloat16}`; widen the result
back to f32 for anything that follows (a softmax sum, say).
"""
@noinline function vexp(v::Vec{N, T}) where {N, T}
    return Base.inferencebarrier(v)::Vec{N, T}
end

"""
    vreduce_add(v) -> T

Sum the lanes of `v`, as one `vector.reduction`.
"""
@noinline function vreduce_add(v::Vec{N, T}) where {N, T}
    return Base.inferencebarrier(zero(T))::T
end

for op in (:vadd, :vsub, :vmul, :vdiv, :vmax, :vmin)
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

# --- matrix intrinsics -------------------------------------------------------
# The `Mat` counterparts of `vload`/`vstore!` (a 2-D `vector.load`/`vector.store`) plus
# the matrix-multiply the vector unit accelerates.

"""
    vload(Mat{R,C,T}, tile, i, j) -> Mat{R,C,T}

Read the `R`x`C` block of `tile` at `(i, j)` as one rank-2 `vector.load`.
"""
@noinline function vload(::Type{Mat{R, C, T}}, tile::Tile, I::Int...) where {R, C, T}
    return Base.inferencebarrier(tile)::Mat{R, C, T}
end

"""
    vstore!(m::Mat, tile, i, j)

Write the `R`x`C` matrix `m` to `tile` at `(i, j)` as one rank-2 `vector.store`.
"""
@noinline function vstore!(m::Mat{R, C, T}, tile::Tile, I::Int...) where {R, C, T}
    Base.donotdelete(m, tile, I)
    return nothing
end

"""
    vmatmul(a::Mat{R,K}, b::Mat{K,C}, acc::Mat{R,C}) -> Mat{R,C}

`a * b + acc` on the AIE matrix-multiply unit, as one `aievec.matmul`. This is the
shaped matmul that keeps the MAC array saturated -- far more MACs per load than the
scalar-broadcast `vfma`. On AIE2P the supported bf16 shape is `4`x`8` * `8`x`4` -> `4`x`4`
accumulating in f32; the operands are widened bf16, the accumulator/result f32.
"""
@noinline function vmatmul(a::Mat{R, K, T}, b::Mat{K, C, T}, acc::Mat{R, C, S}) where {R, K, C, T, S}
    return Base.inferencebarrier(acc)::Mat{R, C, S}
end

# The surface a kernel writes. These inline away, leaving the intrinsics; because
# `Vec` is IRON's own type, they are ordinary methods rather than overlays.
Base.:+(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vadd(a, b)
Base.:-(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vsub(a, b)
Base.:*(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vmul(a, b)
Base.:/(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vdiv(a, b)
Base.muladd(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N, T} = vfma(a, b, c)
Base.fma(a::Vec{N, T}, b::Vec{N, T}, c::Vec{N, T}) where {N, T} = vfma(a, b, c)
Base.max(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vmax(a, b)
Base.min(a::Vec{N, T}, b::Vec{N, T}) where {N, T} = vmin(a, b)
Base.reinterpret(::Type{S}, v::Vec{N, T}) where {N, S, T} = vreinterpret(Vec{N, S}, v)
Base.sum(v::Vec{N, T}) where {N, T} = vreduce_add(v)
Base.exp(v::Vec{N, T}) where {N, T} = vexp(v)
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
