using IRON
using IRON: Tile, compile_kernel!, context, memref_type, objectfifo_type, region
using MLIR: IR
using MLIR.Dialects: func
using BFloat16s: BFloat16
using DLFP8Types: Float8_E4M3FN, Float8_E5M2
using Test

const Buf = Tile{Int32, Tuple{1024}}

# Kernels shared by the matmul testsets.
function matmul!(
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

# bf16 operands widened to an f32 accumulator. `_MM_COMBOS` in
# aie/iron/kernels/linalg.py lists every type combination the accelerator can
# multiply, and (bfloat16, float32) is one of them, while f32 x f32 is not.
function matmul_mixed!(
        a::Tile{BFloat16, Tuple{M, K}}, b::Tile{BFloat16, Tuple{K, N}},
        c::Tile{Float32, Tuple{M, N}},
    ) where {M, K, N}
    for i in 1:M, j in 1:N
        acc = zero(Float32)
        for k in 1:K
            acc += Float32(a[i, k]) * Float32(b[k, j])
        end
        c[i, j] = acc
    end
    return nothing
end

# FP8 operands widened to an f32 accumulator, the split the hardware wants.
function matmul_fp8!(
        a::Tile{F, Tuple{M, K}}, b::Tile{F, Tuple{K, N}}, c::Tile{Float32, Tuple{M, N}}
    ) where {F, M, K, N}
    for i in 1:M, j in 1:N
        acc = zero(Float32)
        for k in 1:K
            acc += Float32(a[i, k]) * Float32(b[k, j])
        end
        c[i, j] = acc
    end
    return nothing
end

# One output row per vector: broadcast an element of `a`, read the matching row of
# `b`, multiply-accumulate. The shape aie::mmul has.
#
# The operands are widened as vectors, never as scalars: both MAC patterns require
# "widening ops in the lhs and rhs operands", so a scalar widened before the
# broadcast leaves the multiply in the accumulator's type, which no hardware
# multiplier has. `Vec{N,T}(::Vec{N,T})` emits nothing, so one kernel covers the
# same-type case too.
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

# Too few subscripts for the tile's rank.
function bad_rank(a::Tile{Float32, Tuple{4, 4}}, b::Tile{Float32, Tuple{4, 4}})
    for i in 1:4
        b[i, i] = a[i]
    end
    return nothing
end

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

    @testset "NPUArray host array" begin
        # Built directly from a placeholder buffer: this exercises the Julia-side
        # array interface and the Adapt mapping without an NPU or the Python stack.
        a = IRON.NPUArray{Int32, 2}(IRON.Py(nothing), (2, 3))
        @test a isa IRON.AbstractGPUArray{Int32, 2}
        @test eltype(a) === Int32
        @test size(a) == (2, 3)
        @test ndims(a) == 2
        @test length(a) == 6

        # A host buffer adapts to the kernel-side Tile view it will be compiled
        # against -- the IRON analogue of CuArray -> CuDeviceArray.
        @test IRON.kernelconvert(a) === Tile{Int32, Tuple{2, 3}}

        # Scalar access is guarded off by default (assertscalar fires before the
        # buffer is touched); under @allowscalar it forwards to the buffer, which
        # here is only a placeholder, so it fails past the guard rather than at it.
        @test_throws Exception a[1, 1]
        @test hasmethod(getindex, Tuple{typeof(a), Int, Int})
        @test hasmethod(setindex!, Tuple{typeof(a), Int32, Int, Int})

        # Contents are never printed (that would copy the buffer back element by
        # element); the summary names the type and shape instead.
        @test occursin("NPUArray", sprint(show, a))
        @test occursin("2×3", sprint(show, MIME("text/plain"), a))
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

    @testset "matmul over element types" begin
        # One generic kernel, instantiated per element type: the arithmetic op and
        # its MLIR type follow the tile's element type with no per-type code.
        cases = (
            Int32 => ("arith.muli", "arith.addi", "i32"),
            Float16 => ("arith.mulf", "arith.addf", "f16"),
            BFloat16 => ("arith.mulf", "arith.addf", "bf16"),
            Float32 => ("arith.mulf", "arith.addf", "f32"),
            Float64 => ("arith.mulf", "arith.addf", "f64"),
        )
        for (T, (mul, add, mlir_ty)) in cases
            S = Tile{T, Tuple{4, 4}}
            ir = lower(matmul!, Tuple{S, S, S})
            @test occursin("memref<4x4x$mlir_ty>", ir)
            @test occursin("$mul %", ir)
            @test occursin("$add %", ir)
            # The accumulator is carried by the k loop rather than through memory.
            @test occursin(Regex("scf\\.for.*iter_args.*-> \\($mlir_ty"), ir)
        end
    end

    @testset "multi-dimensional indexing" begin
        ir = lower(
            matmul!, Tuple{
                Tile{Float32, Tuple{4, 8}}, Tile{Float32, Tuple{8, 2}}, Tile{Float32, Tuple{4, 2}},
            }
        )
        # Two subscripts per access, and the non-square shapes stay distinct.
        @test occursin(r"memref\.load %\w+\[%\w+, %\w+\] : memref<4x8xf32>", ir)
        @test occursin(r"memref\.load %\w+\[%\w+, %\w+\] : memref<8x2xf32>", ir)
        # The stored value is `%N#0`: the accumulator comes out of the k loop, which
        # carries more than one result.
        @test occursin(r"memref\.store %[\w#]+, %\w+\[%\w+, %\w+\] : memref<4x2xf32>", ir)

        @test_throws "takes 2 subscripts, got 1" lower(
            bad_rank, Tuple{Tile{Float32, Tuple{4, 4}}, Tile{Float32, Tuple{4, 4}}}
        )
    end

    @testset "FP8 formats" begin
        ctx = context()
        @test string(IRON.mlir_eltype(ctx, Float8_E4M3FN)) == "f8E4M3FN"
        @test string(IRON.mlir_eltype(ctx, Float8_E5M2)) == "f8E5M2"

        for F in (Float8_E4M3FN, Float8_E5M2)
            ir = lower(
                matmul_fp8!, Tuple{
                    Tile{F, Tuple{2, 2}}, Tile{F, Tuple{2, 2}}, Tile{Float32, Tuple{2, 2}},
                }
            )
            mlir_ty = string(IRON.mlir_eltype(ctx, F))
            @test occursin("memref<2x2x$mlir_ty>", ir)

            # Each operand widens in exactly one op. Without the overlay method
            # table, inference sees the package's software conversion instead and
            # buries this under a few hundred integer ops.
            @test count("arith.extf", ir) == 2
            @test occursin("arith.extf %", ir)
            @test occursin(" $mlir_ty to f32", ir)
            @test !occursin("arith.bitcast", ir)

            # FP8 is storage only: the arithmetic happens in f32.
            @test occursin("arith.mulf %", ir)
            @test occursin(": f32", ir)
        end
    end

    @testset "kernels compute the right answer" begin
        include("jit.jl")

        # add_one, run rather than read.
        a = Int32.(1:1024)
        b = zeros(Int32, 1024)
        run_kernel(add_one, Tuple{Buf, Buf}, a, b)
        @test b == a .+ Int32(1)

        # matmul: asymmetric operands, neither one the identity, so a transposed
        # tile or a swapped operand pair changes the answer. Exact in f32.
        S = Tile{Float32, Tuple{8, 8}}
        x = Float32[10i + j for i in 1:8, j in 1:8]
        y = Float32[i - 2j for i in 1:8, j in 1:8]
        z = zeros(Float32, 8, 8)
        run_kernel(matmul!, Tuple{S, S, S}, x, y, z)
        @test z == x * y

        # Non-square, so a swapped pair of subscripts cannot go unnoticed.
        A, B, C = Tile{Float32, Tuple{4, 8}}, Tile{Float32, Tuple{8, 2}}, Tile{Float32, Tuple{4, 2}}
        x = Float32[i + 2j for i in 1:4, j in 1:8]
        y = Float32[3i - j for i in 1:8, j in 1:2]
        z = zeros(Float32, 4, 2)
        run_kernel(matmul!, Tuple{A, B, C}, x, y, z)
        @test z == x * y

        # bf16 operands into an f32 accumulator: the mixed precision the hardware
        # multiplies. Every value is a small integer, exact in bf16 and in f32.
        Abf = Tile{BFloat16, Tuple{4, 4}}
        Cf = Tile{Float32, Tuple{4, 4}}
        p = BFloat16[i + j for i in 1:4, j in 1:4]
        q = BFloat16[i - 2j for i in 1:4, j in 1:4]
        r = zeros(Float32, 4, 4)
        run_kernel(matmul_mixed!, Tuple{Abf, Abf, Cf}, p, q, r)
        @test r == Float32.(p) * Float32.(q)
    end

    @testset "vector kernels" begin
        # 16 lanes: convert-vector-to-aievec lowers vector.fma only for f32 at 16
        # lanes and bf16 at 16 or 32, matching AIE2's 512-bit vector registers.
        S = Tile{Float32, Tuple{16, 16}}
        ir = lower(matmul_vec!, Tuple{S, S, S})

        # The vector dialect is what convert-vector-to-aievec ingests; these four
        # ops are the whole interface to the AIE vector unit.
        @test occursin("vector.broadcast %", ir)
        @test occursin(r"vector\.load .*vector<16xf32>", ir)
        @test occursin(r"vector\.fma .*: vector<16xf32>", ir)
        @test occursin(r"vector\.store .*vector<16xf32>", ir)

        # The accumulator stays in a vector register across the k loop.
        @test occursin(r"scf\.for.*iter_args.*-> \(vector<16xf32>", ir)

        # They must be registered ops, not unregistered ones that happen to print:
        # the generic form is the tell, and it reaches aiecc as something no
        # pattern matches.
        @test !occursin("\"vector.fma\"", ir)

        # bf16 in, f32 accumulator: both fma operands must come from an extf on
        # bf16 or aiecc refuses to legalize the vector.fma.
        A = Tile{BFloat16, Tuple{16, 16}}
        C = Tile{Float32, Tuple{16, 16}}
        bf = lower(matmul_vec!, Tuple{A, A, C})
        @test occursin(r"vector\.load .*memref<16x16xbf16>, vector<16xbf16>", bf)
        @test count("arith.extf", bf) == 2
        @test occursin("arith.extf %16 : vector<16xbf16> to vector<16xf32>", bf) ||
            occursin(r"arith\.extf .*: vector<16xbf16> to vector<16xf32>", bf)
        @test occursin(r"vector\.fma .*: vector<16xf32>", bf)
        @test occursin(r"vector\.store .*vector<16xf32>", bf)

        # vector.fma is floating-point only, so integers spell the MAC out.
        I = Tile{Int32, Tuple{16, 16}}
        int_ir = lower(matmul_vec!, Tuple{I, I, I})
        @test !occursin("vector.fma", int_ir)
        @test occursin(r"arith\.muli .*: vector<16xi32>", int_ir)
        @test occursin(r"arith\.addi .*: vector<16xi32>", int_ir)

        # i16 -> i32 widens as vectors. Widening the scalar before the broadcast
        # would leave an i32 multiply, which no pattern matches and peano rejects
        # with "unable to legalize <16 x s32> G_MUL".
        I16 = Tile{Int16, Tuple{16, 16}}
        I32 = Tile{Int32, Tuple{16, 16}}
        mixed = lower(matmul_vec!, Tuple{I16, I16, I32})
        @test occursin(r"vector\.broadcast .*: i16 to vector<16xi16>", mixed)
        @test count("arith.extsi", mixed) == 2
        @test occursin(r"arith\.extsi .*: vector<16xi16> to vector<16xi32>", mixed)
        @test !occursin(r"arith\.extsi %\w+ : i16 to i32", mixed)

        # FP8 goes through bf16, since the f32 fma wants operands extended from it.
        F = Tile{Float8_E4M3FN, Tuple{16, 16}}
        fp8 = lower(matmul_vec!, Tuple{F, F, C})
        @test occursin(r"arith\.extf .*: vector<16xf8E4M3FN> to vector<16xbf16>", fp8)
        @test occursin(r"arith\.extf .*: vector<16xbf16> to vector<16xf32>", fp8)
        @test occursin(r"vector\.fma .*: vector<16xf32>", fp8)
    end

    @testset "vector kernels compute the right answer" begin
        # Run it, rather than trust that vector.fma means what it reads like.
        S = Tile{Float32, Tuple{16, 16}}
        x = Float32[10i + j for i in 1:16, j in 1:16]
        y = Float32[i - 2j for i in 1:16, j in 1:16]
        z = zeros(Float32, 16, 16)
        run_kernel(matmul_vec!, Tuple{S, S, S}, x, y, z)
        @test z == x * y

        I = Tile{Int32, Tuple{16, 16}}
        p = Int32[10i + j for i in 1:16, j in 1:16]
        q = Int32[i - 2j for i in 1:16, j in 1:16]
        r = zeros(Int32, 16, 16)
        run_kernel(matmul_vec!, Tuple{I, I, I}, p, q, r)
        @test r == p * q

        # The bf16 -> f32 kernel: values are small integers, exact in bf16.
        A = Tile{BFloat16, Tuple{16, 16}}
        C = Tile{Float32, Tuple{16, 16}}
        u = BFloat16[(i + j) % 7 for i in 1:16, j in 1:16]
        v = BFloat16[(i - 2j) % 5 for i in 1:16, j in 1:16]
        w = zeros(Float32, 16, 16)
        run_kernel(matmul_vec!, Tuple{A, A, C}, u, v, w)
        @test w == Float32.(u) * Float32.(v)

        # The FP8 vector kernel is checked as MLIR above but cannot be run here:
        # lowering `vector<16xf8E4M3FN>` to LLVM asks x86's DataLayout for the
        # alignment of an FP8 vector, which it has no answer for, and MLIR 18
        # crashes rather than reporting it. The scalar FP8 kernels below do run.
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
