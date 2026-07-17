# The dataflow frontend: object FIFOs, workers and a runtime sequence, assembled
# into an `aie.device` module. Mirrors the structure of the Python IRON API.

"""
    ObjectFifo{T<:Tile}(name; depth=2)

A circular buffer carrying tiles of type `T` from one producer endpoint to one
consumer endpoint. Obtain endpoints with [`producer`](@ref) and [`consumer`](@ref).
"""
struct ObjectFifo{T <: Tile}
    name::String
    depth::Int
end

ObjectFifo{T}(name::AbstractString; depth::Integer = 2) where {T <: Tile} =
    ObjectFifo{T}(String(name), Int(depth))

tiletype(::ObjectFifo{T}) where {T} = T

"""
    Endpoint

One end of an [`ObjectFifo`](@ref), as passed to a [`Worker`](@ref) or a runtime
DMA. `port` records which end it is.
"""
struct Endpoint{T <: Tile}
    fifo::ObjectFifo{T}
    port::ObjectFifoPort
end

"""
    producer(fifo) -> Endpoint

The producing end of `fifo`: whoever holds it writes tiles into the FIFO.
"""
producer(fifo::ObjectFifo{T}) where {T} = Endpoint{T}(fifo, Produce)

"""
    consumer(fifo) -> Endpoint

The consuming end of `fifo`: whoever holds it reads tiles out of the FIFO.
"""
consumer(fifo::ObjectFifo{T}) where {T} = Endpoint{T}(fifo, Consume)

tiletype(::Endpoint{T}) where {T} = T

"""
    Worker(kernel, endpoints; stack_size=1024)

A program for one compute tile. On every iteration it acquires one tile from each
endpoint, calls `kernel` with them, and releases them.

`kernel` is a Julia function taking one [`Tile`](@ref) per endpoint, in order, and
returning `nothing`; it is compiled to MLIR and inlined into the core body.

`stack_size` is the core's stack, in bytes; the default matches Python IRON's. The
stack shares the tile's memory with the object FIFO buffers, so a kernel that
spills more than it reserves corrupts them rather than failing outright.
"""
struct Worker{F}
    kernel::F
    endpoints::Vector{Endpoint}
    stack_size::Int
end

Worker(kernel, endpoints::AbstractVector; stack_size::Integer = 1024) =
    Worker(kernel, Vector{Endpoint}(endpoints), Int(stack_size))

"""
    Runtime()

The host-side DMA program. Populate it with [`fill!`](@ref) and [`drain!`](@ref),
then hand it to a [`Program`](@ref).
"""
struct Runtime
    workers::Vector{Worker}
    transfers::Vector{Any}
end

Runtime() = Runtime(Worker[], Any[])

"""
    start!(rt, worker)

Register `worker` to run on the device for the lifetime of the sequence.
"""
start!(rt::Runtime, w::Worker) = (push!(rt.workers, w); rt)

# A DMA between a host buffer (identified by its index among the sequence
# arguments) and a FIFO endpoint. `wait` makes the sequence block on completion.
struct Transfer
    endpoint::Endpoint
    arg::Int
    wait::Bool
end

"""
    fill!(rt, endpoint, arg)

Stream host buffer `arg` (a 1-based index into the sequence arguments) into
`endpoint`.
"""
Base.fill!(rt::Runtime, ep::Endpoint, arg::Integer) =
    (push!(rt.transfers, Transfer(ep, Int(arg), false)); rt)

"""
    drain!(rt, endpoint, arg; wait=true)

Stream `endpoint` out to host buffer `arg` (a 1-based index into the sequence
arguments). With `wait=true` the sequence blocks until the transfer completes,
which is what makes the result visible to the host.
"""
drain!(rt::Runtime, ep::Endpoint, arg::Integer; wait::Bool = true) =
    (push!(rt.transfers, Transfer(ep, Int(arg), wait)); rt)

"""
    Program(device, runtime, argtypes; name="main")

A complete design. `argtypes` gives the [`Tile`](@ref) type of each host buffer
passed to the runtime sequence. Call [`generate_mlir`](@ref) to emit it.
"""
struct Program
    device::AIEDevice
    runtime::Runtime
    argtypes::Vector{Type}
    name::String
end

Program(device::AIEDevice, rt::Runtime, argtypes::AbstractVector; name::AbstractString = "main") =
    Program(device, rt, Vector{Type}(argtypes), String(name))

# Collect every FIFO the design mentions, ordered by name for deterministic output.
function fifos(p::Program)
    found = Dict{String, ObjectFifo}()
    for w in p.runtime.workers, ep in w.endpoints
        found[ep.fifo.name] = ep.fifo
    end
    for t in p.runtime.transfers
        found[t.endpoint.fifo.name] = t.endpoint.fifo
    end
    return [found[k] for k in sort!(collect(keys(found)))]
end

# The core body: acquire one tile per endpoint, run the kernel, release. The whole
# thing sits in an unbounded scf.for, so the core loops forever and is paced by
# FIFO backpressure.
function emit_core_body!(ctx::IR.Context, w::Worker)
    body = IR.Block(IR.Type[], IR.Location[])

    lower = arith.constant(;
        value = IR.Attribute(0, IR.IndexType(; context = ctx)), location = loc(ctx)
    )
    upper = arith.constant(;
        value = IR.Attribute(typemax(Int), IR.IndexType(; context = ctx)), location = loc(ctx)
    )
    step = arith.constant(;
        value = IR.Attribute(1, IR.IndexType(; context = ctx)), location = loc(ctx)
    )
    push!(body, lower)
    push!(body, upper)
    push!(body, step)

    loop_body = IR.Block([IR.IndexType(; context = ctx)], [loc(ctx)])
    tiles = IR.Value[]
    for ep in w.endpoints
        T = tiletype(ep)
        acquire = objectfifo_acquire_op(
            ctx, ep.fifo.name, ep.port, 1, objectfifo_subview_type(ctx, T)
        )
        push!(loop_body, acquire)
        access = objectfifo_subview_access_op(
            ctx, IR.result(acquire, 1), 0, memref_type(ctx, T)
        )
        push!(loop_body, access)
        push!(tiles, IR.result(access, 1))
    end

    argtypes = Tuple{map(tiletype, w.endpoints)...}
    compile_kernel!(ctx, loop_body, w.kernel, argtypes, tiles)

    for ep in w.endpoints
        push!(loop_body, objectfifo_release_op(ctx, ep.fifo.name, ep.port, 1))
    end
    push!(loop_body, scf.yield(IR.Value[]; location = loc(ctx)))

    push!(
        body, scf.for_(
            IR.result(lower, 1), IR.result(upper, 1), IR.result(step, 1), IR.Value[];
            region = region(loop_body), results = IR.Type[], location = loc(ctx),
        )
    )
    push!(body, end_op(ctx))

    return region(body)
end

# A contiguous transfer of the whole tile, expressed as the 4-dimensional access
# pattern the hardware buffer descriptors expect: the three outer dimensions are
# degenerate and the innermost walks the buffer.
function transfer_dims(T::Type{<:Tile})
    len = prod(size(T))
    return Tuple{Int, Int}[(1, 0), (1, 0), (1, 0), (len, 1)], len
end

function emit_runtime_sequence!(ctx::IR.Context, p::Program)
    arg_types = IR.Type[memref_type(ctx, T) for T in p.argtypes]
    body = IR.Block(arg_types, [loc(ctx) for _ in arg_types])

    # Configure and start each transfer in turn, so they are all in flight before
    # the sequence blocks on any of them.
    tasks = Tuple{Transfer, IR.Operation}[]
    for t in p.runtime.transfers
        dims, len = transfer_dims(tiletype(t.endpoint))

        bd_body = IR.Block(IR.Type[], IR.Location[])
        push!(bd_body, dma_bd_op(ctx, IR.argument(body, t.arg), dims, len))
        push!(bd_body, end_op(ctx))

        # A task only needs a completion token when the sequence waits on it.
        task = dma_configure_task_for_op(
            ctx, t.endpoint.fifo.name, region(bd_body); issue_token = t.wait,
        )
        push!(body, task)
        push!(body, dma_start_task_op(ctx, IR.result(task, 1)))
        push!(tasks, (t, task))
    end

    # Await the tasks that asked for it, then release the ones that did not:
    # awaiting a task also frees it, so freeing it again would be a double free.
    for (t, task) in tasks
        t.wait && push!(body, dma_await_task_op(ctx, IR.result(task, 1)))
    end
    for (t, task) in tasks
        t.wait || push!(body, dma_free_task_op(ctx, IR.result(task, 1)))
    end

    return runtime_sequence_op(ctx, "sequence", region(body))
end

# Tile placement here assumes the shape of the reference design: a single worker,
# with every FIFO running between the host and that one core. Anything else needs a
# real placement story, so say so plainly instead of emitting a subtly wrong module.
function check_supported(p::Program)
    length(p.runtime.workers) == 1 || error(
        "IRON: expected exactly one worker, got $(length(p.runtime.workers)); \
        multi-worker designs are not supported yet"
    )
    for fifo in fifos(p)
        n = count(t -> t.endpoint.fifo.name == fifo.name, p.runtime.transfers)
        n == 1 || error(
            "IRON: object FIFO \"$(fifo.name)\" has $n host transfers, expected exactly \
            one; core-to-core FIFOs are not supported yet"
        )
    end
    return nothing
end

"""
    generate_mlir(program; canonicalize=true, ctx=IRON.context()) -> String

Compile `program` -- including its Julia kernels -- to MLIR and return the module
as text, ready for `aie-opt`/`aiecc`.

`canonicalize` tidies the emitted kernel, which mostly means dropping the loop
carries that structurizing Julia's IR leaves behind; see [`canonicalize!`](@ref).
Pass `false` to see exactly what the kernel compiler emitted.

The upstream ops print with their custom assembly, while the `aie` ops print in
generic form because this context does not have the dialect registered. `aie-opt`,
which does, accepts either.
"""
function generate_mlir(p::Program; canonicalize::Bool = true, ctx::IR.Context = context())
    check_supported(p)
    device_body = IR.Block(IR.Type[], IR.Location[])

    # One compute tile per worker, plus a shim tile per host transfer.
    core_tiles = IR.Value[]
    for _ in p.runtime.workers
        tile = logical_tile_op(ctx, CoreTile)
        push!(device_body, tile)
        push!(core_tiles, IR.result(tile, 1))
    end
    shim_tiles = Dict{String, IR.Value}()
    for t in p.runtime.transfers
        tile = logical_tile_op(ctx, ShimNOCTile)
        push!(device_body, tile)
        shim_tiles[t.endpoint.fifo.name] = IR.result(tile, 1)
    end

    # A FIFO runs shim->core when the host fills it and core->shim when the host
    # drains it, so the shim sits at whichever end the host is not.
    for fifo in fifos(p)
        shim = shim_tiles[fifo.name]
        core = only(core_tiles)
        host_port = only(
            t.endpoint.port for t in p.runtime.transfers if t.endpoint.fifo.name == fifo.name
        )
        producer, consumer = host_port === Produce ? (shim, core) : (core, shim)
        push!(
            device_body, objectfifo_op(
                ctx, fifo.name, producer, IR.Value[consumer],
                objectfifo_type(ctx, tiletype(fifo)), fifo.depth,
            )
        )
    end

    for (w, tile) in zip(p.runtime.workers, core_tiles)
        push!(
            device_body,
            core_op(ctx, tile, emit_core_body!(ctx, w); stack_size = w.stack_size),
        )
    end

    push!(device_body, emit_runtime_sequence!(ctx, p))
    push!(device_body, end_op(ctx))

    mod = IR.Module(loc(ctx))
    push!(IR.body(mod), device_op(ctx, p.device, p.name, region(device_body)))

    # Verification only covers the upstream ops -- the aie ops are unregistered here
    # and carry no verifier -- but that is enough to catch a malformed kernel before
    # it reaches aiecc, where the same mistake surfaces far less legibly.
    IR.verify(IR.Operation(mod)) || error(
        "IRON: generated an invalid MLIR module (see the diagnostics above)"
    )
    canonicalize && canonicalize!(mod, ctx)

    return string(IR.Operation(mod))
end
