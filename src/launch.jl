# A high-level launch front end: describe a single-core design by the kernel and
# the buffers it runs on, and let IRON wire up the object FIFOs, the worker, the
# host DMA and the compile/run cycle -- the way `@cuda kernel(a, b, c)` hides the
# grid/stream machinery of a CUDA launch.
#
# This sits on top of the `Program`/`Worker`/`Runtime` layer in `dataflow.jl`. It
# targets a single compute tile with one host transfer stream per buffer. Two shapes
# are supported:
#
#   * whole-buffer (the default): each NPUArray moves as a single tile, the kernel
#     runs once -- the elementwise/whole-array case (`add_one`).
#   * tiled *map*: each NPUArray splits into a grid of tiles, streamed one at a time,
#     and the kernel runs once per tile. All buffers must split into the *same* grid,
#     so the k-th tile of each corresponds; the kernel sees one tile from each buffer
#     per step. This is the SIMT-in-spirit case -- write the per-tile kernel, and the
#     launch drives it over the tile grid, much as a CUDA grid runs a thread block per
#     chunk.
#
# What this call form deliberately does NOT cover is a *reduction* like GEMM, where the
# output tile is held across an inner loop that streams the input operands at a
# different rate and accumulates -- and where each operand's tile is indexed by a
# different combination of the loop variables. That schedule cannot be read off the
# kernel; it is spelled out with `@iron`'s `for` form (see `schedule.jl`).
#
# Unlike a GPU, where every argument is just resident memory the kernel reads and
# writes at will, an AIE buffer streams through a *unidirectional* object FIFO: it
# is either fed host->core (an input the kernel reads) or drained core->host (an
# output the kernel writes). That direction is the one thing the front end cannot
# guess, so each argument is tagged with [`In`](@ref) or [`Out`](@ref) at the call
# site. The tile shape, when a buffer is streamed in more than one piece, is given by
# a type annotation -- `In(a)::Tile{T,Dims}` -- read like a typeassert.

"""
    In(a::NPUArray)

Mark `a` as an **input** to an [`@iron`](@ref) launch: its contents are streamed
host->core and the kernel reads the corresponding tile. Annotate a tile shape with
`In(a)::Tile{T,Dims}` to stream `a` in tiles of that shape rather than whole. See
also [`Out`](@ref).
"""
struct In{A <: NPUArray}
    array::A
    tile::Union{Nothing, DataType}
end
In(a::NPUArray) = In(a, nothing)

"""
    Out(a::NPUArray)

Mark `a` as an **output** of an [`@iron`](@ref) launch: the kernel writes the
corresponding tile and the result is streamed core->host into `a`, ready to read
back with `Array(a)`. Annotate a tile shape with `Out(a)::Tile{T,Dims}` to stream
`a` out in tiles of that shape rather than whole. See also [`In`](@ref).
"""
struct Out{A <: NPUArray}
    array::A
    tile::Union{Nothing, DataType}
end
Out(a::NPUArray) = Out(a, nothing)

const IOArg = Union{In, Out}

_array(x::In) = x.array
_array(x::Out) = x.array
_dir(::In) = :in
_dir(::Out) = :out

# The tile the kernel sees for this argument: the explicit annotation if given, else
# the whole buffer (a one-tile stream).
_tile_of(x::In) = x.tile === nothing ? kernelconvert(x.array) : x.tile
_tile_of(x::Out) = x.tile === nothing ? kernelconvert(x.array) : x.tile

# `In(a)::T` / `Out(a)::T` reach the launch already constructed, so the annotation is
# folded into a fresh wrapper carrying the tile type. `T` must be a `Tile{...}`.
_with_tile(x::In, ::Type{T}) where {T <: Tile} = In(x.array, T)
_with_tile(x::Out, ::Type{T}) where {T <: Tile} = Out(x.array, T)
_with_tile(x, ::Type{T}) where {T} = error(
    "IRON: a tile annotation must be a `Tile{T,Dims}` on an `In(...)`/`Out(...)` \
    argument, as in `In(a)::Tile{Int32,Tuple{1024}}`; got `$(typeof(x))::$T`."
)

# A launch argument must carry its direction; a bare NPUArray is the common mistake,
# so name the fix rather than letting `_dir`/`_array` fail on a MethodError.
_array(x) = error(
    "IRON: @iron arguments must be wrapped in `In(...)` or `Out(...)` to give their \
    streaming direction; got a $(typeof(x)). For example `@iron kernel(In(a), Out(b))`."
)
_dir(x) = _array(x)  # reuse the same message

# --- tile grid geometry ------------------------------------------------------
# Pure arithmetic over shapes: how a column-major buffer splits into tiles, and the
# 4-dimensional hardware buffer-descriptor pattern that gathers one tile.

# The grid a buffer splits into: one factor per dimension. Errors unless the tile
# divides the buffer evenly, since a partial tile has no descriptor here.
function _tile_grid(buffer::Type{<:Tile}, tile::Type{<:Tile})
    bd, td = size(buffer), size(tile)
    length(bd) == length(td) || error(
        "IRON: tile $tile and buffer $buffer have different rank ($(length(td)) vs $(length(bd)))"
    )
    all(bd .% td .== 0) || error(
        "IRON: tile $(td) must divide buffer $(bd) evenly in every dimension"
    )
    return bd .÷ td
end

# The tile positions, 0-based, in column-major order (first axis fastest) -- the same
# order every buffer is walked in, so the k-th position names the k-th tile of each.
_grid_coords(grid::Dims) = [Tuple(c) .- 1 for c in CartesianIndices(grid)]

# The (offset, dims, len) buffer-descriptor access pattern that gathers the tile at
# 0-based grid position `g` from a column-major buffer. `dims` is the four (size,
# stride) pairs the hardware walks, innermost last; the two leading `(1, 0)` pad to
# four dimensions. A `Tile` is column-major, so the innermost dimension steps one
# element down a column and the next steps the buffer's leading extent across columns.
function _tile_pattern(bufdims::NTuple{1, Int}, tdims::NTuple{1, Int}, g::NTuple{1, Int})
    t = tdims[1]
    return g[1] * t, Tuple{Int, Int}[(1, 0), (1, 0), (1, 0), (t, 1)], t
end
function _tile_pattern(bufdims::NTuple{2, Int}, tdims::NTuple{2, Int}, g::NTuple{2, Int})
    R = bufdims[1]
    tr, tc = tdims
    ti, tj = g
    offset = ti * tr + tj * tc * R
    return offset, Tuple{Int, Int}[(1, 0), (1, 0), (tc, R), (tr, 1)], tr * tc
end
_tile_pattern(bufdims::NTuple{N}, _, _) where {N} = error(
    "IRON: @iron streams 1-D and 2-D buffers in tiles; got a $(N)-D buffer. A design \
    that tiles a higher-rank buffer needs a hand-written `Program`."
)

# --- design assembly ---------------------------------------------------------

# One compiled design per (kernel, buffer types, tile types, directions, device, name,
# flags), so a repeated launch reuses the xclbin and its XRT context instead of
# recompiling. The CompiledProgram also caches the launch context after the first run
# (see runtime.jl), so a hot loop of `@iron` calls compiles and opens the device once.
const _LAUNCH_CACHE = Dict{Any, CompiledProgram}()

# The whole-buffer design (one tile per buffer): defer to the dataflow layer directly.
function _build_program(@nospecialize(kernel), dirs, arrays, device, name)
    rt = Runtime()
    worker_endpoints = Endpoint[]
    fifos = ObjectFifo[]
    for (i, arr) in enumerate(arrays)
        fifo = ObjectFifo{kernelconvert(arr)}("arg$i")
        push!(fifos, fifo)
        push!(worker_endpoints, dirs[i] === :in ? consumer(fifo) : producer(fifo))
    end
    start!(rt, Worker(kernel, worker_endpoints))
    for (i, _) in enumerate(arrays)
        if dirs[i] === :in
            fill!(rt, producer(fifos[i]), i)
        else
            drain!(rt, consumer(fifos[i]), i)
        end
    end
    return Program(device, rt, Type[kernelconvert(a) for a in arrays]; name)
end

# The host DMA for a tiled map: for each tile position, in the shared grid order,
# start the input transfers that feed the core and the output transfers that drain
# it, then wait for the outputs before moving on -- bounding the in-flight descriptors
# to one tile, the same discipline the `for`-form reduction uses per output tile.
function _emit_tiled_runtime!(ctx::IR.Context, dirs, buffer_types, tile_types, coords)
    arg_types = IR.Type[memref_type(ctx, T) for T in buffer_types]
    body = IR.Block(arg_types, [loc(ctx) for _ in arg_types])
    args = IR.Value[IR.argument(body, i) for i in eachindex(buffer_types)]

    function tile_task(i, coord; token)
        offset, dims, len = _tile_pattern(size(buffer_types[i]), size(tile_types[i]), coord)
        bd = IR.Block(IR.Type[], IR.Location[])
        push!(bd, dma_bd_op(ctx, args[i], dims, len; offset))
        push!(bd, end_op(ctx))
        task = dma_configure_task_for_op(ctx, "arg$i", region(bd); issue_token = token)
        push!(body, task)
        push!(body, dma_start_task_op(ctx, IR.result(task, 1)))
        return IR.result(task, 1)
    end

    for coord in coords
        pending, outs = IR.Value[], IR.Value[]
        for i in eachindex(dirs)
            if dirs[i] === :in
                push!(pending, tile_task(i, coord; token = false))
            else
                push!(outs, tile_task(i, coord; token = true))
            end
        end
        # Await the outputs (a `dma_await_task` also frees the task, so only a token
        # task may be awaited), then free the tokenless inputs. At least one output is
        # guaranteed by the check in `_iron_launch`.
        for o in outs
            push!(body, dma_await_task_op(ctx, o))
        end
        for p in pending
            push!(body, dma_free_task_op(ctx, p))
        end
    end
    return runtime_sequence_op(ctx, "sequence", region(body))
end

# The tiled-map design. The core side is exactly the whole-buffer core -- an unbounded
# acquire/kernel/release loop over one tile per endpoint (`emit_core_body!`) -- since
# a core already streams tile by tile; only the host DMA tiles the buffers.
function _build_tiled_program(
        @nospecialize(kernel), dirs, buffer_types, tile_types, device, name;
        depth::Integer = 2, stack_size::Integer = 1024, ctx::IR.Context = context(),
    )
    nargs = length(dirs)
    grids = map(_tile_grid, buffer_types, tile_types)
    allequal(grids) || error(
        "IRON: @iron tiling needs every buffer to split into the same tile grid, got $grids"
    )
    coords = _grid_coords(first(grids))

    device_body = IR.Block(IR.Type[], IR.Location[])
    core = logical_tile_op(ctx, CoreTile)
    push!(device_body, core)
    core_tile = IR.result(core, 1)
    shims = IR.Value[]
    for _ in 1:nargs
        t = logical_tile_op(ctx, ShimNOCTile)
        push!(device_body, t)
        push!(shims, IR.result(t, 1))
    end

    endpoints = Endpoint[]
    for i in 1:nargs
        T = tile_types[i]
        prod, cons = dirs[i] === :in ? (shims[i], core_tile) : (core_tile, shims[i])
        push!(device_body, objectfifo_op(ctx, "arg$i", prod, IR.Value[cons], objectfifo_type(ctx, T), depth))
        fifo = ObjectFifo{T}("arg$i")
        push!(endpoints, dirs[i] === :in ? consumer(fifo) : producer(fifo))
    end

    worker = Worker(kernel, endpoints; stack_size)
    push!(device_body, core_op(ctx, core_tile, emit_core_body!(ctx, worker); stack_size))
    push!(device_body, _emit_tiled_runtime!(ctx, dirs, buffer_types, tile_types, coords))
    push!(device_body, end_op(ctx))

    mod = IR.Module(loc(ctx))
    push!(IR.body(mod), device_op(ctx, device, name, region(device_body)))
    IR.verify(IR.Operation(mod)) ||
        error("IRON: @iron generated an invalid MLIR module (see the diagnostics above)")
    canonicalize!(mod, ctx)
    return string(IR.Operation(mod))
end

# The runtime behind `@iron`. Splits the direction-tagged arguments, picks the
# whole-buffer or tiled path, compiles once per distinct shape, and runs it on the
# buffers in place.
function _iron_launch(
    @nospecialize(kernel), args::Tuple;
    device::AIEDevice = npu2,
    name::AbstractString = "main",
    flags::AbstractVector{<:AbstractString} = String[],
    verbose::Bool = false,
    stack_size::Integer = 1024,
)
    dirs = map(_dir, args)
    arrays = map(_array, args)
    buffer_types = map(kernelconvert, arrays)   # the whole-buffer tile type per arg
    tile_types = map(_tile_of, args)            # what the kernel sees per step

    any(d === :out for d in dirs) || error(
        "IRON: @iron needs at least one `Out(...)` argument; a design's results can \
        only leave the core through an output stream, not an `In` buffer."
    )
    for (bt, tt) in zip(buffer_types, tile_types)
        eltype(bt) === eltype(tt) || error(
            "IRON: tile $tt and its buffer (eltype $(eltype(bt))) have different element types"
        )
    end

    key = (typeof(kernel), buffer_types, tile_types, dirs, device, String(name), Tuple(flags), Int(stack_size))
    compiled = get!(_LAUNCH_CACHE, key) do
        if all(tile_types .=== buffer_types)
            compile(_build_program(kernel, dirs, arrays, device, name); flags, verbose)
        else
            mlir = _build_tiled_program(kernel, dirs, buffer_types, tile_types, device, name; stack_size)
            compile(mlir, length(args); flags, verbose)
        end
    end

    run!(compiled, arrays...)
    return compiled
end

"""
    @iron [option = value...] kernel(In(a), Out(b), ...)
    @iron [option = value...] for <space axes>
        @init init_kernel(outputs...)
        @reduce for <reduce axes>
            step_kernel(In(a)[axes...], Out(c)[axes...], ...)
        end
    end

Compile and run a single-core NPU design straight from a kernel and the buffers it
operates on, wiring up the object FIFOs, the worker and the host DMA automatically
-- the IRON analogue of `@cuda kernel(a, b, c)`.

## Call form -- elementwise / tiled map

`kernel` is an ordinary Julia function (see [`Worker`](@ref)); each argument is an
[`NPUArray`](@ref) tagged with its streaming direction, [`In`](@ref) for a buffer
the kernel reads or [`Out`](@ref) for one it writes.

By default a whole buffer moves as one tile and the kernel runs once. To stream a
buffer in pieces, annotate the **tile shape** like a typeassert:

    @iron add_one(In(a)::Tile{Int32,Tuple{1024}}, Out(b)::Tile{Int32,Tuple{1024}})

Then each buffer splits into a grid of tiles, the kernel runs once per tile, and the
tiles stream through one at a time -- write the per-tile kernel and the launch drives
it over the grid, SIMT in spirit. Every annotated buffer must split into the *same*
grid, so the k-th tile of each corresponds.

## `for` form -- tiled reductions (e.g. GEMM)

A reduction accumulates an output tile across an inner loop that streams the inputs
at a different rate -- a schedule that cannot be read off the kernels, so it is
written as a loop nest: the outer `for` header is the **output-tile iteration** (the
space axes), and a nested `@reduce for` is the accumulation loop.

  * `@init k(outputs...)` names a kernel run on the `Out`/accumulator tiles once per
    output tile (e.g. zeroing); omit it for none.
  * `@reduce for <axes> ... end` declares the reduction axes in its header, and its
    body is the **step** call -- the per-step kernel with every operand written
    `In(a)[axes...]` or `Out(c)[axes...]`, where the bracketed axes are the tile access
    (which loop variables index that buffer, one per dimension) and the argument order
    is the kernel's. For a pure per-tile map (no reduction), write the step call
    directly in the space body with no `@reduce`.

A tile shape is inferred from the buffer and the extents of the axes indexing it, so
none is annotated.

## Common behaviour

Inputs are flushed to the device before the launch and outputs read back after, so
once the call returns an `Out` buffer's result is ready via `Array`. The compiled
design is cached (by kernel, buffer/tile types, directions/axes, device and flags),
so a repeated launch neither recompiles nor reopens the device.

Options, written `key = value` before the call or `for`, pass through to compilation:
`device` (an [`AIEDevice`](@ref), default `npu2`), `flags` (extra `aiecc` flags),
`verbose`, `name`, and `stack_size` (the core stack in bytes; reductions often need
more than the 1024-byte default).

```julia
# map: b = a + 1, streamed in 1024-element tiles
@iron add_one(In(a)::Tile{Int32,Tuple{1024}}, Out(b)::Tile{Int32,Tuple{1024}})

# reduction: C = A * B in (m,k) x (k,n) tiles reduced on one core
@iron stack_size = 3328 flags = ["--alloc-scheme=basic-sequential"] for mi in 1:div(M, m), nj in 1:div(N, n)
    @init gemm_zero!(C)
    @reduce for kk in 1:div(K, k)
        gemm_acc!(In(A)[mi, kk], In(B)[kk, nj], Out(C)[mi, nj])
    end
end
```

Returns the [`CompiledProgram`](@ref); the results live in the `Out` buffers.
"""
macro iron(exprs...)
    isempty(exprs) && error(
        "@iron: expected a kernel call `kernel(In(a), Out(b))` or a `for` schedule"
    )
    target = exprs[end]

    kws = Expr[]
    for opt in exprs[1:(end - 1)]
        (Meta.isexpr(opt, :(=)) && opt.args[1] isa Symbol) ||
            error("@iron: options must be written `key = value`, got `$opt`")
        push!(kws, Expr(:kw, opt.args[1], esc(opt.args[2])))
    end

    # A `for` target is a tiled reduction schedule; anything else must be a call.
    Meta.isexpr(target, :for) && return _build_iron_schedule(target, kws)
    Meta.isexpr(target, :call) || error(
        "@iron: expected a kernel call `kernel(In(a), Out(b))` or a `for` schedule, got `$target`"
    )

    kernel = target.args[1]
    # An argument is either `In(a)`/`Out(b)` or an annotated `In(a)::Tile{...}`; the
    # annotation folds the tile type into the wrapper via `_with_tile`.
    kernel_args = map(target.args[2:end]) do a
        if Meta.isexpr(a, :(::))
            Expr(:call, _with_tile, esc(a.args[1]), esc(a.args[2]))
        else
            esc(a)
        end
    end
    return Expr(:call, _iron_launch, esc(kernel), Expr(:tuple, kernel_args...), kws...)
end
