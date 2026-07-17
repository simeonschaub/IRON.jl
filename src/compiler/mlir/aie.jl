# Constructors for the `aie` and `aiex` operations used by a dataflow design.
# Enum values mirror include/aie/Dialect/AIE/IR/AIEAttrs.td.

"""
    AIEDevice

Target device. The value is the `device` enum on `aie.device`.
"""
@enum AIEDevice::Int32 begin
    xcvc1902 = 1
    xcve2302 = 2
    xcve2802 = 3
    npu1 = 4
    npu1_1col = 5
    npu1_2col = 6
    npu1_3col = 7
    npu2 = 8
    npu2_1col = 9
    npu2_2col = 10
    npu2_3col = 11
    npu2_4col = 12
    npu2_5col = 13
    npu2_6col = 14
    npu2_7col = 15
end

"""
    TileType

Kind of tile a logical tile is placed on.
"""
@enum TileType::Int32 begin
    CoreTile = 0
    MemTile = 1
    ShimNOCTile = 2
    ShimPLTile = 3
end

"""
    ObjectFifoPort

Which end of an object FIFO a core accesses: `Produce` writes, `Consume` reads.
"""
@enum ObjectFifoPort::Int32 begin
    Produce = 0
    Consume = 1
end

i32(x, ctx) = IR.Attribute(Int32(x), IR.Type(Int32; context = ctx))
loc(ctx) = IR.Location(; context = ctx)

# `aie.device`: the top-level container for a design.
function device_op(ctx, dev::AIEDevice, sym_name::AbstractString, body::IR.Region)
    return create_op(
        "aie.device", loc(ctx);
        regions = [body],
        properties = [
            "device" => i32(Integer(dev), ctx),
            "sym_name" => IR.Attribute(String(sym_name); context = ctx),
        ],
    )
end

# `aie.logical_tile`: an unplaced tile. The --aie-place-tiles pass assigns real
# coordinates and rewrites these into `aie.tile`.
function logical_tile_op(ctx, tile_type::TileType)
    return create_op(
        "aie.logical_tile", loc(ctx);
        results = [IR.IndexType(; context = ctx)],
        properties = ["tile_type" => i32(Integer(tile_type), ctx)],
    )
end

# `aie.objectfifo`: a circular buffer from one producer tile to one or more
# consumer tiles.
function objectfifo_op(
        ctx, sym_name::AbstractString, producer::IR.Value, consumers::Vector{IR.Value},
        elem_type::IR.Type, depth::Integer,
    )
    return create_op(
        "aie.objectfifo", loc(ctx);
        operands = IR.Value[producer, consumers...],
        properties = [
            "dimensionsFromStreamPerConsumer" =>
                opaque_attr("#aie<bd_dim_layout_array_array[[]]>"; context = ctx),
            "dimensionsToStream" => opaque_attr("#aie<bd_dim_layout_array[]>"; context = ctx),
            "disable_synchronization" => IR.Attribute(false; context = ctx),
            "elemNumber" => i32(depth, ctx),
            "elemType" => IR.Attribute(elem_type),
            "plio" => IR.Attribute(false; context = ctx),
            "sym_name" => IR.Attribute(String(sym_name); context = ctx),
            "via_DMA" => IR.Attribute(false; context = ctx),
        ],
    )
end

# `aie.core`: the program running on a compute tile.
function core_op(ctx, tile::IR.Value, body::IR.Region; stack_size::Integer = 1024)
    return create_op(
        "aie.core", loc(ctx);
        operands = IR.Value[tile],
        results = [IR.IndexType(; context = ctx)],
        regions = [body],
        properties = ["stack_size" => i32(stack_size, ctx)],
    )
end

# `aie.objectfifo.acquire`: take `size` objects from one end of a FIFO.
function objectfifo_acquire_op(
        ctx, fifo::AbstractString, port::ObjectFifoPort, size::Integer, subview_type::IR.Type,
    )
    return create_op(
        "aie.objectfifo.acquire", loc(ctx);
        results = [subview_type],
        properties = [
            "objFifo_name" => IR.FlatSymbolRefAttribute(String(fifo); context = ctx),
            "port" => i32(Integer(port), ctx),
            "size" => i32(size, ctx),
        ],
    )
end

# `aie.objectfifo.subview.access`: project one buffer out of an acquired subview.
function objectfifo_subview_access_op(ctx, subview::IR.Value, index::Integer, memref_type::IR.Type)
    return create_op(
        "aie.objectfifo.subview.access", loc(ctx);
        operands = IR.Value[subview],
        results = [memref_type],
        properties = ["index" => i32(index, ctx)],
    )
end

# `aie.objectfifo.release`: hand `size` objects back to the FIFO.
function objectfifo_release_op(ctx, fifo::AbstractString, port::ObjectFifoPort, size::Integer)
    return create_op(
        "aie.objectfifo.release", loc(ctx);
        properties = [
            "objFifo_name" => IR.FlatSymbolRefAttribute(String(fifo); context = ctx),
            "port" => i32(Integer(port), ctx),
            "size" => i32(size, ctx),
        ],
    )
end

# `aie.end`: terminator for aie regions.
end_op(ctx) = create_op("aie.end", loc(ctx))

# `aie.runtime_sequence`: the host-side DMA program; its block arguments are the
# host buffers passed at launch.
function runtime_sequence_op(ctx, sym_name::AbstractString, body::IR.Region)
    return create_op(
        "aie.runtime_sequence", loc(ctx);
        regions = [body],
        properties = ["sym_name" => IR.Attribute(String(sym_name); context = ctx)],
    )
end

# `aiex.dma_configure_task_for`: build a DMA task targeting an objectfifo.
function dma_configure_task_for_op(
        ctx, alloc::AbstractString, body::IR.Region; issue_token::Bool = false,
    )
    properties = Pair{String, IR.Attribute}[
        "alloc" => IR.FlatSymbolRefAttribute(String(alloc); context = ctx),
    ]
    # `issue_token` is a unit-style flag: present means true, absent means false.
    issue_token && push!(properties, "issue_token" => IR.Attribute(true; context = ctx))
    return create_op(
        "aiex.dma_configure_task_for", loc(ctx);
        results = [IR.IndexType(; context = ctx)],
        regions = [body],
        properties,
    )
end

# `aie.dma_bd`: one buffer descriptor. `dims` is a list of (size, stride) pairs,
# outermost first, describing the access pattern over `buffer`.
#
# Since mlir-aie #3306 the op takes offset/len/sizes/strides as SSA operands via a
# DynamicIndexList. IRON only ever needs static values, so they go in the
# `static_*` attributes and the buffer is the sole operand -- hence
# `operandSegmentSizes = [1, 0, 0, 0, 0]` over the (buffer, offset, len, sizes,
# strides) segments. Sizes/strides are emitted only for an actual n-d pattern; a
# plain contiguous transfer is just offset+len.
function dma_bd_op(
        ctx, buffer::IR.Value, dims::Vector{Tuple{Int, Int}}, len::Integer; offset::Integer = 0,
    )
    props = Pair{String, IR.Attribute}[
        "operandSegmentSizes" => opaque_attr("array<i32: 1, 0, 0, 0, 0>"; context = ctx),
        "burst_length" => i32(0, ctx),
        "static_offset" => i32(offset, ctx),
        "static_len" => i32(len, ctx),
    ]
    if !isempty(dims)
        sizes = join((s for (s, _) in dims), ", ")
        strides = join((t for (_, t) in dims), ", ")
        push!(props, "static_sizes" => opaque_attr("array<i64: $sizes>"; context = ctx))
        push!(props, "static_strides" => opaque_attr("array<i64: $strides>"; context = ctx))
    end
    return create_op("aie.dma_bd", loc(ctx); operands = IR.Value[buffer], properties = props)
end

for (jl_name, op_name) in (
        (:dma_start_task_op, "aiex.dma_start_task"),
        (:dma_await_task_op, "aiex.dma_await_task"),
        (:dma_free_task_op, "aiex.dma_free_task"),
    )
    @eval $jl_name(ctx, task::IR.Value) =
        create_op($op_name, loc(ctx); operands = IR.Value[task])
end
