# C = A * B across the NPU's compute-core array -- the whole-array GEMM, written to
# line up with AMD's mlir-aie matrix-multiplication reference design, with ALL of the
# Julia around the kernel compiled to MLIR (no `@iron` DSL):
#
#   * The micro-kernel mirrors `aie_kernels/aie2p/mm.cc`: a generic blocked
#     `matmul_vectorized!` over `aie::mmul`-shaped (r, s, t) blocks (here `vmatmul`,
#     which lowers to `aievec.matmul_aie2p`), instantiated per type combination as
#     `matmul_bf16_f32!` / `matmul_i8_i32!` -- the names mm.cc exports -- plus
#     `zero_vectorized!` from zero.cc.
#   * The core body `gemm_core` mirrors whole_array.py's `core_fn`, but is *ordinary
#     Julia*: the acquires, the zero, the k loop and the releases are written as a
#     function over `Reader`/`Writer` FIFO handles and compiled by the same
#     `compile_kernel!` path that compiles the kernels (see docs/julia-as-mlir.md).
#     Residency and pipelining are where a statement sits in the loop nest -- the
#     output tile is held across the k loop because its acquire is outside it.
#   * `build_design` mirrors whole_array.py's `_build_design`, op for op: the same
#     three-layer objectfifo pyramid (A broadcasts through a MemTile to every core,
#     B distributes through one MemTile per group of 4 cores, C joins back through
#     the same group MemTile), the same block streaming (`dims_to_stream`), the same
#     0xD00 core stack, and a runtime sequence that keeps a FIFO_DEPTH-deep window
#     of DMA tasks in flight so drains overlap the next tile's feed and compute.
#
# Two deliberate deviations from mm.cc, both load-bearing:
#
#   1. Everything is column-major (Julia's convention, mm.cc's is row-major): a
#      streamed tile arrives block-columnar -- an (r*s, num_blocks) L1 tile whose
#      column b is the b-th (r, s) block -- so a whole block is one contiguous
#      `Mat` load and the result needs no transpose: `Array(dC)` really is A * B.
#   2. mm.cc keeps its running accumulators in registers across the k loop (the
#      MMUL C00 objects). Peano v21's AIE2P backend miscompiles loop-carried vector
#      accumulators (crash before the combiner patch, wrong answers after), so the
#      partial sum lives in the C tile instead: load / vmatmul / store per k block.
#      That also drops mm.cc's 2x2 m/n expansion, whose point is accumulator-
#      register utilization. Verified correct on hardware; costs ~one C round-trip
#      per block without moving the headline number (the design is DMA/overhead-
#      bound at scale, ~57 GF/s bf16 / ~63 GOP/s int8 at 512^3 on 32 cores).
#
# Run on an NPU (sizes overridable via GEMM_M/GEMM_K/GEMM_N; N = 512 uses the full
# 32-core array):
#   IRON_RUN=1 julia --project examples/gemm.jl

using IRON
using BFloat16s: BFloat16
using Printf

const IR = IRON.IR
using IRON: Reader, Writer, acquire!, release!
using IRON: CoreTile, MemTile, ShimNOCTile, FIFO_DEPTH,
    logical_tile_op, objectfifo_op, objectfifo_link_op, core_op, runtime_sequence_op,
    dma_await_task_op, dma_free_task_op, end_op, objectfifo_type, memref_type,
    compile_kernel!, finish_module, region, loc, context,
    _dma_task!, _tile_pattern, _dims_to_stream_blocks, _dims_to_stream_unblock

# The per-core tile, cf. mm.cc's DIM_M/DIM_K/DIM_N compile-time defines: each core
# reduces one (DIM_M, DIM_N) output tile over K in (DIM_K)-deep steps. m is grown to
# 128 (not n) so the core count stays N/DIM_N while each core runs fewer, larger
# tiles -- the amortization that makes the MAC-array kernel pay.
const DIM_M, DIM_K, DIM_N = 128, 64, 16

# --- the micro-kernels (cf. mm.cc / zero.cc) ----------------------------------

"""
    zero_vectorized!(c)

Clear an output tile with whole-vector stores, one L1 column at a time -- the
column-major counterpart of zero.cc's `zero_vectorized<T, M, N>`.
"""
function zero_vectorized!(c::Tile{T, Tuple{RT, NB}}) where {T, RT, NB}
    zeros = zero(Vec{RT, T})
    for j in 1:NB
        vstore!(zeros, c, 1, j)
    end
    return nothing
end

"""
    matmul_vectorized!(a, b, c, Val(rowA), Val(colA), Val(colB), Val(r), Val(s), Val(t))

`c += a * b` over block-columnar tiles, the counterpart of mm.cc's
`matmul_vectorized_2x2_mmul<T_in, T_out, rowA, colA, colB, r, s, t>`: `a` is
rowA x colA blocks of (r, s), `b` is colA x colB blocks of (s, t), `c` is
rowA x colB blocks of (r, t), and each block is one `vmatmul` (the `aie::mmul`).
Block b is column b of its L1 tile; the orders match mm.cc's row-major pointer
walk -- A (z outer, i inner), B (i outer, j inner), C (z outer, j inner).

The k accumulation carries the running (r, t) partial in the C tile (load /
accumulate / store per block) rather than in a register, sidestepping the Peano
v21 loop-carried-accumulator bug -- see the file header.
"""
@inline function matmul_vectorized!(
        a::Tile{T_in}, b::Tile{T_in}, c::Tile{T_out},
        ::Val{rowA}, ::Val{colA}, ::Val{colB}, ::Val{r}, ::Val{s}, ::Val{t},
    ) where {T_in, T_out, rowA, colA, colB, r, s, t}
    for z in 0:(rowA - 1), j in 0:(colB - 1)
        for i in 0:(colA - 1)
            acc = vload(Mat{r, t, T_out}, c, 1, z * colB + j + 1)
            acc = vmatmul(
                vload(Mat{r, s, T_in}, a, 1, z * colA + i + 1),
                vload(Mat{s, t, T_in}, b, 1, i * colB + j + 1),
                acc,
            )
            vstore!(acc, c, 1, z * colB + j + 1)
        end
    end
    return nothing
end

# The per-type instantiations, named as mm.cc exports them, and a `matmul!` front
# that picks one by dispatch -- Julia's version of mm.cc's `combos` table. The
# (r, s, t) block is the hardware matmul shape: bf16 is 4x8 * 8x4 -> 4x4 into f32
# (128 MACs/op), int8 is 8x8 * 8x8 -> 8x8 into i32 (512 MACs/op).
matmul_bf16_f32!(a::Tile{BFloat16}, b::Tile{BFloat16}, c::Tile{Float32}) =
    matmul_vectorized!(
        a, b, c,
        Val(DIM_M ÷ 4), Val(DIM_K ÷ 8), Val(DIM_N ÷ 4), Val(4), Val(8), Val(4),
    )
matmul_i8_i32!(a::Tile{Int8}, b::Tile{Int8}, c::Tile{Int32}) =
    matmul_vectorized!(
        a, b, c,
        Val(DIM_M ÷ 8), Val(DIM_K ÷ 8), Val(DIM_N ÷ 8), Val(8), Val(8), Val(8),
    )

matmul!(a::Tile{BFloat16}, b::Tile{BFloat16}, c::Tile{Float32}) = matmul_bf16_f32!(a, b, c)
matmul!(a::Tile{Int8}, b::Tile{Int8}, c::Tile{Int32}) = matmul_i8_i32!(a, b, c)

# --- the core body (cf. whole_array.py's core_fn) -----------------------------

"""
    gemm_core(a, b, c, Val(KT))

One compute core's whole program, as ordinary Julia: forever, acquire an output
tile, zero it, stream `KT` pairs of input tiles through it accumulating, release
it. Compiled to the core's MLIR by `compile_kernel!` -- the acquires become
`aie.objectfifo.acquire`, the loops `scf.for`, the kernels inline vector code.

The FIFO wiring is the *signature*: each `Reader`/`Writer` names its FIFO in its
type, so instantiating this one function per core against that core's FIFO names
is what places it. The outer loop runs forever (each launch feeds a finite number
of tiles and the core blocks on the next acquire) so a re-launch is simply more
tiles.
"""
function gemm_core(a::Reader, b::Reader, c::Writer, ::Val{KT}) where {KT}
    while true
        ct = acquire!(c)
        zero_vectorized!(ct)
        for _ in 1:KT
            at = acquire!(a)
            bt = acquire!(b)
            matmul!(at, bt, ct)
            release!(a)
            release!(b)
        end
        release!(c)
    end
end

# --- the design (cf. whole_array.py's _build_design) --------------------------

# Cores come in groups of up to 4 (npu2's 4 compute rows per column), one MemTile
# per group carrying that group's B distribute and C join.
function core_groups(ncores)
    ncores <= 4 && return [collect(0:(ncores - 1))]
    ncores % 4 == 0 || error("beyond 4 cores the count must be a multiple of 4")
    return [collect(g:(g + 3)) for g in 0:4:(ncores - 1)]
end

"""
    build_design(Tin, Tacc, (r, s, t), M, K, N) -> String

Assemble the whole `aie.device` as MLIR text: the tiles, the three-layer object-
FIFO pyramid, one compiled `gemm_core` per compute core, and the host runtime
sequence. Ordinary Julia is the metaprogramming layer -- the loops below are the
structure whole_array.py spells out with ObjectFifo split/forward/join.
"""
function build_design(::Type{Tin}, ::Type{Tacc}, (r, s, t), M, K, N) where {Tin, Tacc}
    (M % DIM_M == 0 && K % DIM_K == 0 && N % DIM_N == 0) ||
        error("(M, K, N) = ($M, $K, $N) must be multiples of ($DIM_M, $DIM_K, $DIM_N)")
    ncores = N ÷ DIM_N
    groups = core_groups(ncores)
    length(groups) <= 8 || error("more than npu2's 8 memtile columns; use fewer cores")

    # The L1 (core-side) tile of each operand is block-columnar: each (r, s) block
    # one contiguous column, `nb` blocks. What the FIFOs carry.
    AT = Tile{Tin, Tuple{r * s, (DIM_M ÷ r) * (DIM_K ÷ s)}}
    BT = Tile{Tin, Tuple{s * t, (DIM_K ÷ s) * (DIM_N ÷ t)}}
    CT = Tile{Tacc, Tuple{r * t, (DIM_M ÷ r) * (DIM_N ÷ t)}}

    ctx = context()
    body = IR.Block(IR.Type[], IR.Location[])
    tile!(t) = (op = logical_tile_op(ctx, t); push!(body, op); IR.result(op, 1))
    fifo!(name, prod, cons, T; dims = Tuple{Int, Int}[]) = push!(
        body, objectfifo_op(ctx, name, prod, cons, objectfifo_type(ctx, T), FIFO_DEPTH; dims_to_stream = dims)
    )

    core_tiles = [tile!(CoreTile) for _ in 1:ncores]

    # Per group: one MemTile and one shim column. B's group super-tile (DIM_K,
    # 4*DIM_N) comes in whole and the link's dst_offsets slice one (DIM_K, DIM_N)
    # tile per core, re-blocked for the kernel by `dims_to_stream` on each l2l1; C
    # joins the per-core block-columnar tiles and the l2l3's `dims_to_stream`
    # un-blocks them back to DDR column-major order.
    group_mems = IR.Value[]
    for (gi, group) in enumerate(groups)
        g, gsize = gi - 1, length(group)
        mem, shim = tile!(MemTile), tile!(ShimNOCTile)
        push!(group_mems, mem)

        fifo!("b_l3l2_g$g", shim, [mem], Tile{Tin, Tuple{DIM_K, gsize * DIM_N}})
        for c in group
            fifo!("b_l2l1_c$c", mem, [core_tiles[c + 1]], BT;
                dims = _dims_to_stream_blocks(DIM_K, DIM_N, s, t))
        end
        push!(body, objectfifo_link_op(ctx, ["b_l3l2_g$g"], ["b_l2l1_c$c" for c in group];
            dst_offsets = [(c - group[1]) * DIM_N * DIM_K for c in group]))

        fifo!("c_l2l3_g$g", mem, [shim], Tile{Tacc, Tuple{DIM_M, gsize * DIM_N}};
            dims = _dims_to_stream_unblock(DIM_M, DIM_N, r, t))
        for c in group
            fifo!("c_l1l2_c$c", core_tiles[c + 1], [mem], CT)
        end
        push!(body, objectfifo_link_op(ctx, ["c_l1l2_c$c" for c in group], ["c_l2l3_g$g"];
            src_offsets = [(c - group[1]) * DIM_N * DIM_M for c in group]))
    end

    # A broadcasts to every core through one MemTile: across DDR once, fanned out
    # on-chip. It gets its own MemTile while columns are free and folds onto a
    # group's once the groups use all 8 (the 32-core case).
    a_shim = tile!(ShimNOCTile)
    a_mem = length(groups) < 8 ? tile!(MemTile) : group_mems[1]
    fifo!("a_l3l2", a_shim, [a_mem], Tile{Tin, Tuple{DIM_M, DIM_K}})
    fifo!("a_l2l1", a_mem, core_tiles, AT; dims = _dims_to_stream_blocks(DIM_M, DIM_K, r, s))
    push!(body, objectfifo_link_op(ctx, ["a_l3l2"], ["a_l2l1"]))

    # One compiled `gemm_core` per core, wired to its FIFOs through its argument
    # types. 3328 = 0xD00, whole_array.py's stack_size.
    for c in 0:(ncores - 1)
        argtypes = Tuple{
            Reader{:a_l2l1, AT}, Reader{Symbol("b_l2l1_c$c"), BT},
            Writer{Symbol("c_l1l2_c$c"), CT}, Val{K ÷ DIM_K},
        }
        core = IR.Block(IR.Type[], IR.Location[])
        compile_kernel!(ctx, core, gemm_core, argtypes, IR.Value[])
        push!(core, end_op(ctx))
        push!(body, core_op(ctx, core_tiles[c + 1], region(core); stack_size = 3328))
    end

    push!(body, build_sequence(ctx, Tin, Tacc, M, K, N, groups))
    push!(body, end_op(ctx))
    return finish_module(ctx, npu2, "main", body; what = "the GEMM design ")
end

# The host DMA program (cf. whole_array.py's rt.fill/rt.drain loops): for each
# output-tile row, feed every core its reduction inputs in lockstep -- A's shared
# tile once per k step, B's group super-tiles once per group -- and start draining
# each group's C super-tile. Each FIFO keeps a FIFO_DEPTH-deep window of tasks in
# flight, retiring the oldest as a newer one is queued, so one tile's output drain
# overlaps the next tile's feed and compute; the epilogue awaits the tail.
function build_sequence(ctx, Tin, Tacc, M, K, N, groups)
    bufs = (Tile{Tin, Tuple{M, K}}, Tile{Tin, Tuple{K, N}}, Tile{Tacc, Tuple{M, N}})
    body = IR.Block([memref_type(ctx, T) for T in bufs], [loc(ctx) for _ in bufs])
    args = IR.Value[IR.argument(body, i) for i in 1:3]

    task(i, fname, tdims, grid) = _dma_task!(
        ctx, body, args[i], fname, _tile_pattern(size(bufs[i]), tdims, grid)...; token = true
    )
    retire(t) = (push!(body, dma_await_task_op(ctx, t)); push!(body, dma_free_task_op(ctx, t)))
    window = Dict{String, Vector{IR.Value}}()
    inflight!(fname, t) = begin
        q = get!(window, fname, IR.Value[])
        length(q) >= FIFO_DEPTH && retire(popfirst!(q))
        push!(q, t)
    end

    gsize = length(groups[1])
    for mi in 0:(M ÷ DIM_M - 1)
        for kk in 0:(K ÷ DIM_K - 1)
            inflight!("a_l3l2", task(1, "a_l3l2", (DIM_M, DIM_K), (mi, kk)))
            for gi in eachindex(groups)
                g = gi - 1
                inflight!("b_l3l2_g$g", task(2, "b_l3l2_g$g", (DIM_K, gsize * DIM_N), (kk, g)))
            end
        end
        for gi in eachindex(groups)
            g = gi - 1
            inflight!("c_l2l3_g$g", task(3, "c_l2l3_g$g", (DIM_M, gsize * DIM_N), (mi, g)))
        end
    end
    for q in values(window), t in q
        retire(t)
    end
    return runtime_sequence_op(ctx, "sequence", region(body))
end

# --- host driver --------------------------------------------------------------

# Bank-aware allocation (the default) silently overlaps object-FIFO buffers; every
# matrix multiply under programming_examples/ passes this flag.
const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]

# Compile + verify + time one type combination. Timing is end to end (host sync +
# NPU run + sync), the number a caller sees. Operands are small integers, exact in
# bf16/int8, so the product is checked for equality against the host reference.
function run_case(label, ::Type{T_in}, ::Type{T_out}, rst, M, K, N; trials = 10) where {T_in, T_out}
    compiled = IRON.compile(
        build_design(T_in, T_out, rst, M, K, N), [:in, :in, :out]; flags = AIECC_FLAGS
    )
    a = T_in[(i + j) % 7 for i in 1:M, j in 1:K]
    b = T_in[(i - 2j) % 5 for i in 1:K, j in 1:N]
    dA, dB = NPUArray(a), NPUArray(b)
    dC = NPUArray{T_out}(undef, Tile{T_out, Tuple{M, N}})

    IRON.run!(compiled, dA, dB, dC)
    ok = Array(dC) == T_out.(a) * T_out.(b)
    times = [@elapsed IRON.run!(compiled, dA, dB, dC) for _ in 1:trials]
    gops = 2.0 * M * N * K / minimum(times) / 1e9

    @printf("%-12s  %5dx%d x %5dx%d  %2d cores  %-4s  %7.2f GOP/s\n",
            label, M, K, K, N, div(N, DIM_N), ok ? "ok" : "BAD", gops)
    # Free the design's hardware context so the next case doesn't exhaust the NPU's
    # compute columns.
    IRON.release!(compiled)
    return ok
end

if get(ENV, "IRON_RUN", "0") == "1"
    M = parse(Int, get(ENV, "GEMM_M", "512"))
    K = parse(Int, get(ENV, "GEMM_K", "512"))
    N = parse(Int, get(ENV, "GEMM_N", "128"))
    run_case("bf16 -> f32", BFloat16, Float32, (4, 8, 4), M, K, N)
    run_case("i8 -> i32", Int8, Int32, (8, 8, 8), M, K, N)
else
    # Without an NPU, still exercise the whole front end: build both designs and
    # round-trip the MLIR through a parse + verify.
    for (label, Tin, Tacc, rst) in
        (("bf16 -> f32", BFloat16, Float32, (4, 8, 4)), ("i8 -> i32", Int8, Int32, (8, 8, 8)))
        mlir = build_design(Tin, Tacc, rst, 256, 128, 32)
        mod = parse(IR.Module, mlir; context = context())
        println("$label: 256x128 x 128x32, 2 cores, ",
                "verify = $(IR.verify(IR.Operation(mod))), $(count("aie.core", mlir)) core ops")
    end
    println("Run on an NPU with:  IRON_RUN=1 julia --project examples/gemm.jl")
    println("Sizes: GEMM_M/GEMM_K/GEMM_N (defaults 512x512x128; GEMM_N=512 = 32 cores).")
end
