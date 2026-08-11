# Object-FIFO endpoints as seen from *inside* a core body, so that the code around a
# kernel -- the loop nest, the acquires and the releases -- can be written as ordinary
# Julia and compiled by the same path that compiles the kernel itself.
#
# PROTOTYPE. See docs/julia-as-mlir.md.

"""
    FifoRef{Name, Port, T}

A handle to one end of an object FIFO, as a core body sees it. `Name` is the FIFO's
symbol, `Port` is [`Produce`](@ref) or [`Consume`](@ref), and `T` is the [`Tile`](@ref)
it carries.

The name lives in the *type* because `aie.objectfifo.acquire` names its FIFO with a
symbol attribute rather than an operand: the name has to be a compile-time constant, and
a type parameter is a compile-time constant inference can see. A `FifoRef` is therefore a
singleton -- it carries no data and lowers to no MLIR value at all. It exists only so
that a core body can be written as a Julia function whose *signature* records which FIFOs
it is wired to, and be type-inferred against that.
"""
struct FifoRef{Name, Port, T <: Tile} end

const Reader{Name, T} = FifoRef{Name, Consume, T}
const Writer{Name, T} = FifoRef{Name, Produce, T}

fifo_name(::Type{FifoRef{Name, Port, T}}) where {Name, Port, T} = Name
fifo_port(::Type{FifoRef{Name, Port, T}}) where {Name, Port, T} = Port
tiletype(::Type{FifoRef{Name, Port, T}}) where {Name, Port, T} = T

# The intrinsics, following the same discipline as the `vec.jl` ones: `@noinline` so the
# call survives inlining into the IR the compiler walks, and `donotdelete` so Julia treats
# it as effectful -- which is what stops the optimizer from deleting an acquire whose tile
# is unused, or CSE-ing two acquires of the same FIFO into one. Both matter here far more
# than for a load: an acquire and a release are *lock operations*, and duplicating,
# dropping or reordering one deadlocks the array.

"""
    acquire!(fifo) -> Tile

Take one object from `fifo`, blocking until it is available. Lowers to
`aie.objectfifo.acquire` plus the `aie.objectfifo.subview.access` that projects out the
single buffer.

Where the call sits in the loop nest is what makes an operand streaming or stationary:
an acquire inside the per-tile loop re-fetches every iteration, one hoisted out of it
holds the buffer across the whole loop.
"""
@noinline function acquire!(::FifoRef{Name, Port, T}) where {Name, Port, T}
    Base.donotdelete(Name, Port)
    return Base.inferencebarrier(T())::T
end

"""
    release!(fifo)

Hand one object back to `fifo`. Lowers to `aie.objectfifo.release`.
"""
@noinline function release!(::FifoRef{Name, Port, T}) where {Name, Port, T}
    Base.donotdelete(Name, Port)
    return nothing
end
