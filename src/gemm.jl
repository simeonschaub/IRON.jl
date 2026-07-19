# A tiled single-core GEMM: C = A * B, streaming (m,k) x (k,n) tiles through object
# FIFOs and accumulating on-core into an (m,n) tile. A port of the single_core.py
# design from mlir-aie's matrix_multiplication examples, with the micro-kernel
# written in Julia rather than bound from C++.
#
# The Python design leans on machinery IRON.jl does not have yet -- an external
# `mm.cc`, L2 object-FIFO forwarding with `dims_to_stream`, `TensorTiler2D` host
# access patterns, and task-group ping-pong buffering. This keeps the same math and
# the same nested-acquire core loop, but expresses the host transfers as one DMA per
# tile and drives the whole thing from Julia. That is enough for modest tile counts;
# large shapes want the loop/tap mechanism the faithful port would add.

"""
    tile_access(R, tr, tc, ti, tj) -> (offset, dims, len)

The buffer-descriptor access pattern that gathers the `(ti, tj)`-th tile of size
`(tr, tc)` out of a column-major buffer with `R` rows. `ti`/`tj` are 0-based tile
coordinates.

A `Tile` is column-major (see [`memref_type`](@ref)), so a tile is walked column by
column: the innermost dimension steps one element `tr` times (down a column), the
next steps `R` elements `tc` times (across columns) -- `R`, the row count, is the
column stride of a column-major buffer, *not* its column count. `offset` is the
column-major linear index of the tile's top-left element. The two leading `(1, 0)`
dims pad the pattern to the four dimensions the hardware buffer descriptor expects.
"""
function tile_access(R::Int, tr::Int, tc::Int, ti::Int, tj::Int)
    offset = ti * tr + tj * tc * R
    dims = Tuple{Int, Int}[(1, 0), (1, 0), (tc, R), (tr, 1)]
    return offset, dims, tr * tc
end

# The core body of the tiled GEMM. Mirrors single_core.py's `core_fn`:
#
#     for _ in tiles:                 # one (m, n) output tile per iteration
#         c = acquire(outC); zero(c)
#         for _ in K/k:               # reduce over the k dimension
#             a = acquire(inA); b = acquire(inB)
#             matmul(a, b, c)         # c += a * b, accumulating
#             release(inA); release(inB)
#         release(outC)
#
# The zero and matmul kernels are ordinary Julia functions, inlined into place by
# `compile_kernel!` exactly as a `Worker`'s kernel is.
function emit_gemm_core!(
        ctx::IR.Context, zero_kernel, matmul_kernel,
        ::Type{Ta}, ::Type{Tb}, ::Type{Tc}, num_tiles::Int, num_k::Int,
    ) where {Ta <: Tile, Tb <: Tile, Tc <: Tile}
    body = IR.Block(IR.Type[], IR.Location[])
    index = IR.IndexType(; context = ctx)
    const_(v) = (op = arith.constant(; value = IR.Attribute(v, index), location = loc(ctx)); push!(body, op); IR.result(op, 1))
    c0, c1 = const_(0), const_(1)
    ctiles, cks = const_(num_tiles), const_(num_k)

    # Outer loop: one output tile per iteration.
    outer = IR.Block([index], [loc(ctx)])
    cacq = objectfifo_acquire_op(ctx, "outC", Produce, 1, objectfifo_subview_type(ctx, Tc))
    push!(outer, cacq)
    cacc = objectfifo_subview_access_op(ctx, IR.result(cacq, 1), 0, memref_type(ctx, Tc))
    push!(outer, cacc)
    ctile = IR.result(cacc, 1)
    compile_kernel!(ctx, outer, zero_kernel, Tuple{Tc}, IR.Value[ctile])

    # Inner loop: reduce over the k tiles, accumulating into the same C tile.
    inner = IR.Block([index], [loc(ctx)])
    aacq = objectfifo_acquire_op(ctx, "inA", Consume, 1, objectfifo_subview_type(ctx, Ta))
    push!(inner, aacq)
    aacc = objectfifo_subview_access_op(ctx, IR.result(aacq, 1), 0, memref_type(ctx, Ta))
    push!(inner, aacc)
    bacq = objectfifo_acquire_op(ctx, "inB", Consume, 1, objectfifo_subview_type(ctx, Tb))
    push!(inner, bacq)
    bacc = objectfifo_subview_access_op(ctx, IR.result(bacq, 1), 0, memref_type(ctx, Tb))
    push!(inner, bacc)
    compile_kernel!(
        ctx, inner, matmul_kernel, Tuple{Ta, Tb, Tc},
        IR.Value[IR.result(aacc, 1), IR.result(bacc, 1), ctile],
    )
    push!(inner, objectfifo_release_op(ctx, "inA", Consume, 1))
    push!(inner, objectfifo_release_op(ctx, "inB", Consume, 1))
    push!(inner, scf.yield(IR.Value[]; location = loc(ctx)))

    push!(outer, scf.for_(c0, cks, c1, IR.Value[]; region = region(inner), results = IR.Type[], location = loc(ctx)))
    push!(outer, objectfifo_release_op(ctx, "outC", Produce, 1))
    push!(outer, scf.yield(IR.Value[]; location = loc(ctx)))

    push!(body, scf.for_(c0, ctiles, c1, IR.Value[]; region = region(outer), results = IR.Type[], location = loc(ctx)))
    push!(body, end_op(ctx))
    return region(body)
end

# The host DMA program. For each output tile, in row-major (mi, nj) order, stream the
# k tiles of A and B that reduce into it, then drain the finished C tile. Each output
# tile is awaited before the next, which bounds the number of in-flight descriptors
# at the cost of overlap -- the ping-pong the faithful port would add.
function emit_gemm_runtime!(
        ctx::IR.Context, ::Type{Tin}, ::Type{Tacc},
        M::Int, K::Int, N::Int, m::Int, k::Int, n::Int,
    ) where {Tin, Tacc}
    big(T, r, c) = Tile{T, Tuple{r, c}}
    arg_types = IR.Type[memref_type(ctx, big(Tin, M, K)), memref_type(ctx, big(Tin, K, N)), memref_type(ctx, big(Tacc, M, N))]
    body = IR.Block(arg_types, [loc(ctx) for _ in arg_types])
    A, B, C = (IR.argument(body, i) for i in 1:3)

    Mt, Kt, Nt = M ÷ m, K ÷ k, N ÷ n

    # One DMA task feeding `alloc` from `buf`, gathering the given tile. `R` is the
    # row count (column stride) of the column-major buffer `buf`.
    function fill_task(alloc, buf, R, tr, tc, ti, tj; token)
        offset, dims, len = tile_access(R, tr, tc, ti, tj)
        bd = IR.Block(IR.Type[], IR.Location[])
        push!(bd, dma_bd_op(ctx, buf, dims, len; offset))
        push!(bd, end_op(ctx))
        task = dma_configure_task_for_op(ctx, alloc, region(bd); issue_token = token)
        push!(body, task)
        push!(body, dma_start_task_op(ctx, IR.result(task, 1)))
        return IR.result(task, 1)
    end

    for mi in 0:(Mt - 1), nj in 0:(Nt - 1)
        pending = IR.Value[]
        for kk in 0:(Kt - 1)
            # A is M x K (row count M); B is K x N (row count K).
            push!(pending, fill_task("inA", A, M, m, k, mi, kk; token = false))
            push!(pending, fill_task("inB", B, K, k, n, kk, nj; token = false))
        end
        # C is M x N (row count M).
        ctask = fill_task("outC", C, M, m, n, mi, nj; token = true)
        push!(body, dma_await_task_op(ctx, ctask))
        for t in pending
            push!(body, dma_free_task_op(ctx, t))
        end
    end

    return runtime_sequence_op(ctx, "sequence", region(body))
end

"""
    gemm_program(zero_kernel, matmul_kernel, Tin, Tacc, M, K, N, m, k, n;
                 device=npu2, name="main") -> String

Emit the MLIR for a tiled single-core GEMM `C = A * B`, where `A` is `M x K`, `B` is
`K x N` and `C` is `M x N`, all column-major, computed in `(m, k) x (k, n)` tiles
reduced on one core.

`zero_kernel(c::Tile{Tacc,Tuple{m,n}})` clears an output tile and
`matmul_kernel(a::Tile{Tin,Tuple{m,k}}, b::Tile{Tin,Tuple{k,n}}, c::Tile{Tacc,Tuple{m,n}})`
accumulates `c += a * b`; both are ordinary Julia kernels, inlined into the core.

`M`, `K`, `N` must be divisible by `m`, `k`, `n`. The result is ready for
`aie-opt`/`aiecc`, like [`generate_mlir`](@ref).
"""
function gemm_program(
        zero_kernel, matmul_kernel, ::Type{Tin}, ::Type{Tacc},
        M::Integer, K::Integer, N::Integer, m::Integer, k::Integer, n::Integer;
        device::AIEDevice = npu2, name::AbstractString = "main", ctx::IR.Context = context(),
    ) where {Tin, Tacc}
    M % m == 0 && K % k == 0 && N % n == 0 ||
        error("IRON: gemm needs M,K,N divisible by m,k,n; got ($M,$K,$N) over ($m,$k,$n)")
    Ta = Tile{Tin, Tuple{Int(m), Int(k)}}
    Tb = Tile{Tin, Tuple{Int(k), Int(n)}}
    Tc = Tile{Tacc, Tuple{Int(m), Int(n)}}
    num_tiles = (M ÷ m) * (N ÷ n)
    num_k = K ÷ k

    device_body = IR.Block(IR.Type[], IR.Location[])

    core = logical_tile_op(ctx, CoreTile)
    push!(device_body, core)
    core_tile = IR.result(core, 1)
    shims = Dict{String, IR.Value}()
    for name_ in ("inA", "inB", "outC")
        t = logical_tile_op(ctx, ShimNOCTile)
        push!(device_body, t)
        shims[name_] = IR.result(t, 1)
    end

    # inA/inB run shim -> core (host fills), outC runs core -> shim (host drains).
    for (fifo, T, host_produces) in (("inA", Ta, true), ("inB", Tb, true), ("outC", Tc, false))
        prod, cons = host_produces ? (shims[fifo], core_tile) : (core_tile, shims[fifo])
        push!(device_body, objectfifo_op(ctx, fifo, prod, IR.Value[cons], objectfifo_type(ctx, T), 2))
    end

    push!(device_body, core_op(ctx, core_tile, emit_gemm_core!(ctx, zero_kernel, matmul_kernel, Ta, Tb, Tc, num_tiles, num_k); stack_size = 3328))
    push!(device_body, emit_gemm_runtime!(ctx, Tin, Tacc, Int(M), Int(K), Int(N), Int(m), Int(k), Int(n)))
    push!(device_body, end_op(ctx))

    mod = IR.Module(loc(ctx))
    push!(IR.body(mod), device_op(ctx, device, name, region(device_body)))
    IR.verify(IR.Operation(mod)) ||
        error("IRON: gemm generated an invalid MLIR module (see the diagnostics above)")
    canonicalize!(mod, ctx)
    return string(IR.Operation(mod))
end
