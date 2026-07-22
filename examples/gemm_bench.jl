# Benchmark the single-core `@iron for` GEMM (see gemm.jl) across a size sweep and
# report GFLOP/s, so we can hold it up against AMD's mlir-aie matrix_multiplication
# design. The kernels are exactly the ones from gemm.jl; only the driver is new.
#
#   IRON_RUN=1 julia --project examples/gemm_bench.jl
#
# What is measured: end-to-end launch latency -- host->device sync, the NPU run, and
# device->host sync -- as `run!` bundles them (see runtime.jl). That is the honest
# number a caller sees; it is *not* the isolated core compute time, so small sizes are
# dominated by fixed DMA/instruction-stream overhead rather than by the matmul.
#
# To compare against AMD: build mlir-aie's
# programming_examples/basic/matrix_multiplication/whole_array (a multi-core, L2-
# forwarded, aie::mmul design) at the same M/K/N and compare its reported GFLOP/s to
# the column printed here. The gap is analysed in the file footer.

using IRON
using BFloat16s: BFloat16
using Printf
using LinearAlgebra: BLAS

# --- kernels (identical to gemm.jl) ------------------------------------------

# Clear an output tile a column at a time; run once per output tile by `@init`.
function gemm_zero!(c::Tile{Tacc, Tuple{m, n}}) where {Tacc, m, n}
    z = zero(Vec{m, Tacc})
    for j in 1:n
        vstore!(z, c, 1, j)
    end
    return nothing
end

# `c += a * b` for one tile, reading and writing the running accumulator in `c` so a
# sequence of calls reduces over k. Column-major: a column of `a` is vloaded, a scalar
# of `b` broadcast.
function gemm_acc!(
        a::Tile{T, Tuple{m, k}}, b::Tile{T, Tuple{k, n}}, c::Tile{Tacc, Tuple{m, n}},
    ) where {T, Tacc, m, k, n}
    for j in 1:n
        acc = vload(Vec{m, Tacc}, c, 1, j)
        for kk in 1:k
            av = vload(Vec{m, T}, a, 1, kk)
            bv = Vec{m, T}(b[kk, j])
            acc = muladd(Vec{m, Tacc}(av), Vec{m, Tacc}(bv), acc)
        end
        vstore!(acc, c, 1, j)
    end
    return nothing
end

# The hardware matmul-array micro-kernel (`vmatmul`), one 4x8*8x4->4x4 tile. Whole-tile
# `Mat` loads only, so the object-FIFO tile is exactly one matmul (tiny) until
# `dims_to_stream` lands.
function gemm_mm_zero!(c::Tile{Float32, Tuple{4, 4}})
    z = zero(Vec{4, Float32})
    for j in 1:4
        vstore!(z, c, 1, j)
    end
    return nothing
end
function gemm_mm_acc!(
        a::Tile{BFloat16, Tuple{4, 8}}, b::Tile{BFloat16, Tuple{8, 4}},
        c::Tile{Float32, Tuple{4, 4}},
    )
    av = vload(Mat{4, 8, BFloat16}, a, 1, 1)
    bv = vload(Mat{8, 4, BFloat16}, b, 1, 1)
    acc = vload(Mat{4, 4, Float32}, c, 1, 1)
    vstore!(vmatmul(av, bv, acc), c, 1, 1)
    return nothing
end

# The big-tile version: a 128x64 A / 64x16 B / 128x16 C tile, streamed block-columnar via
# `dims_to_stream` in m, k AND n. Grows m to 128 (keeping n=16, so N/16 cores) so each core runs
# 1024 vmatmuls per tile and DMAs far fewer, larger tiles -- the amortization that makes the
# MAC-array kernel compute-bound. The k accumulation is a LOOP with the partial in the C tile
# (load/accumulate/store per k-block), so nothing vector-valued is loop-carried (no Peano PHI).
function gemm_mmt_zero!(c::Tile{Float32, Tuple{16, 128}})
    z = zero(Vec{16, Float32})
    for j in 1:128
        vstore!(z, c, 1, j)
    end
    return nothing
end
function gemm_mmt_acc!(
        a::Tile{BFloat16, Tuple{32, 256}}, b::Tile{BFloat16, Tuple{32, 32}},
        c::Tile{Float32, Tuple{16, 128}},
    )
    for mb in 0:31, nb in 0:3
        bi = mb * 4 + nb
        for kb in 0:7
            acc = vload(Mat{4, 4, Float32}, c, 1, bi + 1)
            acc = vmatmul(vload(Mat{4, 8, BFloat16}, a, 1, mb * 8 + kb + 1), vload(Mat{8, 4, BFloat16}, b, 1, kb * 4 + nb + 1), acc)
            vstore!(acc, c, 1, bi + 1)
        end
    end
    return nothing
end

# The int8 big-tile kernel: AIE2P int8 matmul is 8x8 * 8x8 -> 8x8 (i32 acc, 512 MACs/op, 4x
# the bf16 tile), so blocks are 8x8 and the accumulator is Int32. Same 128x16 big tile and
# PHI-free k-loop as the bf16 kernel; N/16 cores.
function gemm_i8_zero!(c::Tile{Int32, Tuple{64, 32}})
    z = zero(Vec{64, Int32})
    for j in 1:32
        vstore!(z, c, 1, j)
    end
    return nothing
end
function gemm_i8_acc!(
        a::Tile{Int8, Tuple{64, 128}}, b::Tile{Int8, Tuple{64, 16}},
        c::Tile{Int32, Tuple{64, 32}},
    )
    for mb in 0:15, nb in 0:1
        bi = mb * 2 + nb
        for kb in 0:7
            acc = vload(Mat{8, 8, Int32}, c, 1, bi + 1)
            acc = vmatmul(vload(Mat{8, 8, Int8}, a, 1, mb * 8 + kb + 1), vload(Mat{8, 8, Int8}, b, 1, kb * 2 + nb + 1), acc)
            vstore!(acc, c, 1, bi + 1)
        end
    end
    return nothing
end

const AIECC_FLAGS = ["--alloc-scheme=basic-sequential"]
const Tin, Tacc = BFloat16, Float32

# A single launch of the whole GEMM. Wrapped in a function so the `@iron for` reads the
# local sizes; the first call at a given (M,K,N,m,k,n) compiles and caches, later calls
# reuse the cached xclbin (see `_LAUNCH_CACHE` in schedule.jl).
function gemm_launch!(da, db, dc, M, K, N, m, k, n)
    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, m), nj in 1:div(N, n)
        @init gemm_zero!(dc)
        @reduce for kk in 1:div(K, k)
            gemm_acc!(In(da)[mi, kk], In(db)[kk, nj], Out(dc)[mi, nj])
        end
    end
    return nothing
end

# The same GEMM with the output columns spread across the compute-core array via
# `@cores nj` (N/n cores). Beyond ~8 cores this overruns the shim buffer-descriptor
# budget until L2 forwarding lands, so the sweep guards it with the size check below.
function gemm_launch_cores!(da, db, dc, M, K, N, m, k, n)
    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, m), nj in 1:div(N, n)
        @cores nj
        @init gemm_zero!(dc)
        @reduce for kk in 1:div(K, k)
            gemm_acc!(In(da)[mi, kk], In(db)[kk, nj], Out(dc)[mi, nj])
        end
    end
    return nothing
end

# The same multi-core GEMM with every operand routed through a MemTile (`L2(...)`): A
# broadcasts to the cores, B distributes, C joins, so A crosses DDR once instead of once
# per core. distribute/join go through a single memtile, capped at ~5 cores for now.
function gemm_launch_l2!(da, db, dc, M, K, N, m, k, n)
    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, m), nj in 1:div(N, n)
        @cores nj
        @init gemm_zero!(dc)
        @reduce for kk in 1:div(K, k)
            gemm_acc!(L2(In(da))[mi, kk], L2(In(db))[kk, nj], L2(Out(dc))[mi, nj])
        end
    end
    return nothing
end

# The L2 GEMM with the hardware-matmul micro-kernel (`vmatmul`). Fixed 4x8x4 tiles, so the
# core count is N/4 (m,k,n args ignored -- kept for the shared launch signature).
function gemm_launch_mm!(da, db, dc, M, K, N, m, k, n)
    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, 4), nj in 1:div(N, 4)
        @cores nj
        @init gemm_mm_zero!(dc)
        @reduce for kk in 1:div(K, 8)
            gemm_mm_acc!(L2(In(da))[mi, kk], L2(In(db))[kk, nj], L2(Out(dc))[mi, nj])
        end
    end
    return nothing
end

# Big-tile sub-tiled vmatmul: 128x64 A / 64x16 B / 128x16 C operand tiles, all block-columnar,
# with a looped (PHI-free) k accumulation. Grows m to 128 to amortize per-tile overhead while
# holding n=16 so the core count (N/16) matches the scalar-L2 comparison.
function gemm_launch_mmt!(da, db, dc, M, K, N, m, k, n)
    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, 128), nj in 1:div(N, 16)
        @cores nj
        @init gemm_mmt_zero!(dc)
        @reduce for kk in 1:div(K, 64)
            gemm_mmt_acc!(
                L2(In(da); blocks = (4, 8))[mi, kk],
                L2(In(db); blocks = (8, 4))[kk, nj],
                L2(Out(dc); blocks = (4, 4))[mi, nj],
            )
        end
    end
    return nothing
end

# The int8 big-tile GEMM on L2, same 128x16 tile / N/16 cores as gemm_launch_mmt! but 8x8x8
# int8 matmuls (i32 acc). m,k,n args ignored (fixed 128/64/16); kept for the shared signature.
function gemm_launch_i8!(da, db, dc, M, K, N, m, k, n)
    @iron stack_size = 3328 flags = AIECC_FLAGS for mi in 1:div(M, 128), nj in 1:div(N, 16)
        @cores nj
        @init gemm_i8_zero!(dc)
        @reduce for kk in 1:div(K, 64)
            gemm_i8_acc!(
                L2(In(da); blocks = (8, 8))[mi, kk],
                L2(In(db); blocks = (8, 8))[kk, nj],
                L2(Out(dc); blocks = (8, 8))[mi, nj],
            )
        end
    end
    return nothing
end

# The non-L2 scheme uses one shim tile per core, so it fits within the device's shim
# columns (about 8 on npu2). L2 partitions the cores into groups of 4, one memtile each.
const MAX_CORES = 8

# --- one size ----------------------------------------------------------------

# Time `launch` end to end (host sync + NPU run + sync, all bundled by `run!`) over
# `trials`, after one warm-up that also compiles and checks correctness. Returns the
# best GFLOP/s and whether the result matched the host product.
function time_launch(launch, da, db, dc, a, b, M, K, N, m, k, n; trials)
    launch(da, db, dc, M, K, N, m, k, n)            # warm up: compile + one run
    maxerr = maximum(abs.(Array(dc) .- Float32.(a) * Float32.(b)))
    times = Float64[]
    for _ in 1:trials
        push!(times, @elapsed launch(da, db, dc, M, K, N, m, k, n))
    end
    return (; ok = maxerr == 0, maxerr, gflops = 2.0 * M * N * K / minimum(times) / 1e9)
end

function bench_size(M, K, N; m = 16, k = 32, n = 16, trials = 20)
    @assert M % m == 0 && K % k == 0 && N % n == 0 "sizes must be tile multiples"

    # Small integers, exact in bf16, so the reference product can be checked tightly.
    a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
    b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
    da, db = NPUArray(a), NPUArray(b)
    dc = NPUArray{Tacc}(undef, Tile{Tacc, Tuple{M, N}})

    single = time_launch(gemm_launch!, da, db, dc, a, b, M, K, N, m, k, n; trials)

    # Multi-core, only where the core count is within the shim-BD budget.
    ncores = div(N, n)
    multi = if ncores <= MAX_CORES
        try
            time_launch(gemm_launch_cores!, da, db, dc, a, b, M, K, N, m, k, n; trials)
        catch e
            (; ok = false, maxerr = NaN, gflops = NaN, err = sprint(showerror, e))
        end
    else
        (; ok = false, maxerr = NaN, gflops = NaN, err = "$(ncores) cores > $(MAX_CORES) (needs L2)")
    end
    return (; M, K, N, ncores, single, multi)
end

# --- host BLAS reference (a ceiling to read the NPU number against) -----------

function bench_host(M, K, N; trials = 20)
    a = Float32.(BFloat16[(i + j) % 7 for i in 1:M, j in 1:K])
    b = Float32.(BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N])
    c = a * b                       # warm up BLAS
    times = Float64[]
    for _ in 1:trials
        push!(times, @elapsed a * b)
    end
    sort!(times)
    return 2.0 * M * N * K / times[1] / 1e9
end

# --- sweep -------------------------------------------------------------------

if get(ENV, "IRON_RUN", "0") == "1"
    sizes = [128, 512]      # 128 -> 8 cores (the multi-core point); 512 -> single-core ceiling
    # bf16 vs int8 `vmatmul` across the full L2 core array. All three kernels share the L2
    # topology (A broadcasts, B distributes, C joins) and core count (N/16) on *tall* GEMMs
    # (large M·K, small N). N = 64/128/256/512 -> 4/8/16/32 cores. Columns: scalar-broadcast
    # bf16, big-tile bf16 `vmatmul`, big-tile int8 `vmatmul` (8x8x8, 4x MACs/op + half the
    # input bytes). GF/s = GOP/s = 2*M*N*K/t regardless of dtype; the i8/bf16 ratio is the
    # datatype win at scale.
    @printf("%-14s  %5s  %6s  %11s  %11s  %11s  %8s\n",
            "M=K x N", "ok", "cores", "scalar bf16", "vmm bf16", "vmm int8", "i8/bf16")
    println("-"^74)
    # Default keeps the run short; IRON_BENCH_FULL=1 adds the 16- and 32-core designs, whose
    # large MLIR is slow to compile.
    tall = [(512, 64), (512, 128)]
    get(ENV, "IRON_BENCH_FULL", "0") == "1" && append!(tall, [(256, 256), (512, 512)])
    for (s, N) in tall
        M = K = s; m, k, n = 16, 32, 16                  # N/n cores (4, 8, 16, 32)
        ncores = div(N, n)
        try
            a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
            b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
            da, db = NPUArray(a), NPUArray(b)
            dc = NPUArray{Tacc}(undef, Tile{Tacc, Tuple{M, N}})
            l2 = time_launch(gemm_launch_l2!, da, db, dc, a, b, M, K, N, m, k, n; trials = 10)
            mt = time_launch(gemm_launch_mmt!, da, db, dc, a, b, M, K, N, m, k, n; trials = 10)
            # int8 needs its own Int8/Int32 buffers.
            ai = Int8[(i + j) % 7 for i in 1:M, j in 1:K]
            bi = Int8[(i - 2j) % 5 for i in 1:K, j in 1:N]
            dai, dbi = NPUArray(ai), NPUArray(bi)
            dci = NPUArray{Int32}(undef, Tile{Int32, Tuple{M, N}})
            i8 = time_launch(gemm_launch_i8!, dai, dbi, dci, ai, bi, M, K, N, m, k, n; trials = 10)
            @printf("%-14s  %5s  %6d  %11.2f  %11.2f  %11.2f  %7.2fx\n",
                    "$(M)x$(K)x$(N)", (l2.ok && mt.ok && i8.ok) ? "yes" : "NO", ncores,
                    l2.gflops, mt.gflops, i8.gflops, i8.gflops / mt.gflops)
        catch e
            @printf("%-14s  %5s  %s\n", "$(M)x$(K)x$(N)", "ERR", sprint(showerror, e))
        end
    end

    # Large-matrix scaling at the full 32-core array (N=512). Grow M=K to test whether the
    # ~60 GOP/s 32-core wall is fixed per-launch/per-tile overhead (throughput keeps climbing
    # with size) or a shared memtile/DDR resource (plateaus). int8 is the headline number to
    # hold against AMD's ~50 TOPS (int8). GF/s = GOP/s = 2*M*N*K/min-time, end to end.
    if get(ENV, "IRON_BENCH_LARGE", "0") == "1"
        println()
        @printf("%-16s  %5s  %6s  %13s  %13s\n", "M=K x N", "ok", "cores", "bf16 GOP/s", "int8 GOP/s")
        println("-"^64)
        for s in (512, 1024, 2048, 4096, 8192, 16384)
            M = K = s; N = 512; m, k, n = 16, 32, 16       # 32 cores
            try
                a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
                b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
                da, db = NPUArray(a), NPUArray(b)
                dc = NPUArray{Tacc}(undef, Tile{Tacc, Tuple{M, N}})
                mt = time_launch(gemm_launch_mmt!, da, db, dc, a, b, M, K, N, m, k, n; trials = 6)
                ai = Int8[(i + j) % 7 for i in 1:M, j in 1:K]
                bi = Int8[(i - 2j) % 5 for i in 1:K, j in 1:N]
                dai, dbi = NPUArray(ai), NPUArray(bi)
                dci = NPUArray{Int32}(undef, Tile{Int32, Tuple{M, N}})
                i8 = time_launch(gemm_launch_i8!, dai, dbi, dci, ai, bi, M, K, N, m, k, n; trials = 6)
                @printf("%-16s  %5s  %6d  %13.2f  %13.2f\n", "$(M)x$(K)x$(N)",
                        (mt.ok && i8.ok) ? "yes" : "NO", div(N, 16), mt.gflops, i8.gflops)
            catch e
                @printf("%-16s  %5s  %s\n", "$(M)x$(K)x$(N)", "ERR", sprint(showerror, e))
            end
        end
    end

    # The `vmatmul` micro-kernel at three granularities, all on L2, same problem: scalar-
    # broadcast (16x32x16 tiles), tiny vmatmul (4x8x4, one matmul/tile), and the big-tile
    # sub-tiled vmatmul (64x16 output via dims_to_stream in m/k/n, 512 matmuls/tile). The
    # tiny->big span shows the full dims_to_stream + tile-amortization payoff at a fixed 4
    # cores (N=64). Small sizes only -- the tiny column inflates the instruction stream.
    println()
    @printf("%-14s  %12s  %11s  %14s\n",
            "M=K x N", "scalar GF/s", "tiny GF/s", "sub-tiled GF/s")
    println("-"^58)
    for (s, N) in [(128, 64), (256, 64)]
        M = K = s
        try
            a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
            b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
            da, db = NPUArray(a), NPUArray(b)
            dc = NPUArray{Tacc}(undef, Tile{Tacc, Tuple{M, N}})
            sc = time_launch(gemm_launch_l2!, da, db, dc, a, b, M, K, N, 16, 32, 16; trials = 10)
            mm = time_launch(gemm_launch_mm!, da, db, dc, a, b, M, K, N, 4, 8, 4; trials = 10)
            mt = time_launch(gemm_launch_mmt!, da, db, dc, a, b, M, K, N, 16, 32, 16; trials = 10)
            @printf("%-14s  %12.2f  %11.2f  %14.2f\n",
                    "$(M)x$(K)x$(N)", sc.gflops, mm.gflops, mt.gflops)
        catch e
            @printf("%-14s  %8s  %s\n", "$(M)x$(K)x$(N)", "ERR", sprint(showerror, e))
        end
    end

    println()
    println("GFLOP/s = 2*M*N*K / min time. Times are end-to-end (host sync + launch + sync).")
    println("`@cores nj` spreads N/n output columns across that many compute cores;")
    println("beyond $(MAX_CORES) cores it needs L2 forwarding (a later increment).")
    println("BLAS threads: ", BLAS.get_num_threads())
else
    println("GEMM benchmark: single-core vs multi-core (`@cores`) @iron for GEMM.")
    println("Run on an NPU with:  IRON_RUN=1 julia --project=examples examples/gemm_bench.jl")
end

# --- where we lag AMD's mlir-aie GEMM ----------------------------------------
#
# We are closing these one increment at a time; AMD's whole_array example is the
# throughput target. Status, largest gap first:
#
# 1. [in progress] Core array. `@cores nj` now spreads output columns across the
#    compute-core array (this benchmark's N-core column), so we are no longer stuck on
#    one core. The current per-core FIFO scheme caps at ~8 cores before the shim's
#    buffer-descriptor budget is exhausted; lifting that to the full array (32 cores on
#    npu2) is the job of the L2 forwarding increment (gap 4).
#
# 2. Shallow compute/DMA overlap (no real ping-pong). The core reduces ONE output tile
#    to completion before the next (see _emit_schedule_core! in schedule.jl); the host
#    DMA only runs FIFO_DEPTH (=2) input tiles ahead before the object FIFO
#    backpressures. AMD double-buffers whole output tiles with task groups so the DMA of
#    the next tile overlaps compute of the current one across the core array.
#
# 3. Scalar-broadcast microkernel vs aie::mmul. gemm_acc! does a 16-lane column vload
#    of A and broadcasts a *scalar* of B per k-step -- one useful MAC vector per B
#    element. AMD's mm.cc calls the aie::mmul intrinsic (e.g. 4x8x4 / 4x8x8 tiles) that
#    keeps the MAC array saturated. Our inner loop leaves most of the vector-MAC
#    throughput on the table even on the one core we use.
#
# 4. No L2 / TensorTiler streaming. AMD forwards operands through a memtile (L2)
#    objectfifo with `dims_to_stream` and shapes host access with TensorTiler2D, so DDR
#    bandwidth is amortised and reused across cores. We DMA every tile straight from DDR
#    per core, re-reading shared operands.
#
# 5. No m sub-tiling. m is pinned to the 16-lane vector width; mm.cc sub-tiles the row
#    dimension (its `r`) to reuse loaded A/B across several output rows. We reload.
#
# 6. bf16-only operands, f32 accumulate. Fine for this comparison, but AMD's harness
#    also exercises i8/i16 paths with much higher MAC counts per cycle.
#
# 7. Fixed per-launch overhead. Because timing is end-to-end, the small sizes here are
#    dominated by instruction-stream upload + BO sync, not compute -- expect GFLOP/s to
#    climb steeply with size before the single-core compute ceiling flattens it.
