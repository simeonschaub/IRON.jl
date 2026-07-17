using IRON
using IRON: Tile, compile_kernel!, context, memref_type, objectfifo_type, region
using MLIR: IR
using MLIR.Dialects: func
using Test

const Buf = Tile{Int32, Tuple{1024}}

# The design from examples/add_one.jl, which mirrors the Python IRON example that
# the reference generic.mlir was generated from.
function add_one(a::Buf, b::Buf)
    for i in 1:1024
        b[i] = a[i] + Int32(1)
    end
    return nothing
end

returns_value(a::Buf) = 1

# A call with no entry in the lowering table, to check the error rather than a
# silently wrong kernel.
@noinline unknown_op(x::Int32) = Base.inferencebarrier(x)::Int32

function calls_unknown(a::Buf, b::Buf)
    for i in 1:1024
        b[i] = unknown_op(a[i])
    end
    return nothing
end

function add_one_program()
    of_in = ObjectFifo{Buf}("in")
    of_out = ObjectFifo{Buf}("out")
    rt = Runtime()
    start!(rt, Worker(add_one, [consumer(of_in), producer(of_out)]))
    fill!(rt, producer(of_in), 1)
    drain!(rt, consumer(of_out), 2)
    return Program(npu2, rt, [Buf, Buf])
end

# Lower `f` on its own into a `func.func` taking plain memrefs, so a test can cover
# the Julia -> MLIR translation without the surrounding design. The function has to
# be wrapped in a module before printing: MLIR's printer verifies as it goes, and
# verifying an operation that is not inside a region reads a null parent.
function lower(@nospecialize(f), @nospecialize(argtypes))
    ctx = context()
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
    mod = IR.Module(loc)
    push!(IR.body(mod), fn)
    return string(IR.Operation(mod))
end

@testset "IRON.jl" begin
    @testset "tile types" begin
        ctx = context()
        @test string(memref_type(ctx, Buf)) == "memref<1024xi32>"
        @test string(memref_type(ctx, Tile{Float32, Tuple{8, 16}})) == "memref<8x16xf32>"
        @test string(objectfifo_type(ctx, Buf)) == "!aie.objectfifo<memref<1024xi32>>"
        @test size(Buf) == (1024,)
        @test eltype(Buf) === Int32
    end

    @testset "kernel lowering" begin
        ir = lower(add_one, Tuple{Buf, Buf})
        # The loop and the arithmetic survive as ops rather than being folded away:
        # if the load were not opaque to inference, the whole body would vanish.
        @test occursin("scf.for", ir)
        @test occursin("memref.load", ir)
        @test occursin("memref.store", ir)
        @test occursin("arith.addi", ir)

        # The add is on i32, not on the index type of the loop counter. Sharing one
        # constant between the two would be a type error, since Julia considers
        # `Int32(1)` and `1` equal but MLIR does not.
        @test occursin("arith.constant 1 : i32", ir)

        # A dead comparison, left behind by turning the loop's exit test into scf.for
        # bounds, is not emitted.
        @test !occursin("arith.cmpi", ir)
    end

    @testset "unsupported kernels are rejected" begin
        @test_throws "must return nothing" lower(returns_value, Tuple{Buf})
        @test_throws "no lowering registered" lower(calls_unknown, Tuple{Buf, Buf})
    end

    @testset "unsupported designs are rejected" begin
        # Two workers: no placement story yet, so this must not silently emit a
        # module that wires both cores to the same tile.
        two_workers = let rt = Runtime(), f = ObjectFifo{Buf}("in"), g = ObjectFifo{Buf}("out")
            start!(rt, Worker(add_one, [consumer(f), producer(g)]))
            start!(rt, Worker(add_one, [consumer(f), producer(g)]))
            fill!(rt, producer(f), 1)
            drain!(rt, consumer(g), 2)
            Program(npu2, rt, [Buf, Buf])
        end
        @test_throws "exactly one worker" generate_mlir(two_workers)

        # A FIFO the host never touches has no shim to attach to.
        core_to_core = let rt = Runtime(), f = ObjectFifo{Buf}("in"), g = ObjectFifo{Buf}("mid")
            start!(rt, Worker(add_one, [consumer(f), producer(g)]))
            fill!(rt, producer(f), 1)
            Program(npu2, rt, [Buf])
        end
        @test_throws "core-to-core" generate_mlir(core_to_core)
    end

    @testset "generated module" begin
        mlir = generate_mlir(add_one_program())

        # Device and tiles: one core, plus one shim per host transfer.
        @test occursin("device = 8 : i32", mlir)      # npu2
        @test occursin("tile_type = 0 : i32", mlir)   # CoreTile
        @test count("tile_type = 2 : i32", mlir) == 2 # two ShimNOCTiles

        # Both FIFOs exist, and the core sees `in` as a consumer (port 1) and `out`
        # as a producer (port 0).
        @test occursin("sym_name = \"in\"", mlir)
        @test occursin("sym_name = \"out\"", mlir)
        @test occursin("objFifo_name = @in, port = 1 : i32", mlir)
        @test occursin("objFifo_name = @out, port = 0 : i32", mlir)

        # The drained task issues a token and is awaited; the filled one is freed.
        @test occursin("alloc = @out, issue_token = true", mlir)
        @test occursin("aiex.dma_await_task", mlir)
        @test occursin("aiex.dma_free_task", mlir)

        # The kernel is inlined into the core rather than called.
        @test occursin("aie.core", mlir)
        @test occursin("memref.load", mlir)
    end

    @testset "generated module round-trips" begin
        # Re-parsing the text is the closest check available without the toolchain to
        # what aie-opt does first: syntax, types and attributes must all be valid.
        # Note the aie attributes only survive because unregistered dialects are
        # allowed -- they parse as opaque and print back identically.
        mlir = generate_mlir(add_one_program())
        ctx = context()
        mod = parse(IR.Module, mlir; context = ctx)
        @test IR.verify(IR.Operation(mod))
        @test occursin("aie.objectfifo", string(IR.Operation(mod)))
    end
end
