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

    @printf("%-12s  %5s  %11s  %6s  %11s  %8s  %10s\n",
            "M=K=N", "ok", "1-core GF/s", "cores", "N-core GF/s", "speedup", "host BLAS")
    println("-"^76)
    for s in sizes
        try
            r = bench_size(s, s, s)
            hg = bench_host(s, s, s)
            if r.multi.ok
                @printf("%-12s  %5s  %11.2f  %6d  %11.2f  %7.2fx  %10.1f\n",
                        "$(s)³", r.single.ok ? "yes" : "NO",
                        r.single.gflops, r.ncores, r.multi.gflops,
                        r.multi.gflops / r.single.gflops, hg)
            else
                # Multi-core did not run (over the core budget, or a compile/mismatch);
                # still report the single-core number and why the parallel one is absent.
                @printf("%-12s  %5s  %11.2f  %6d  %11s  %8s  %10.1f\n",
                        "$(s)³", r.single.ok ? "yes" : "NO",
                        r.single.gflops, r.ncores, "-", "-", hg)
                @printf("       ^ %d-core: %s\n", r.ncores, get(r.multi, :err, "mismatch"))
            end
        catch e
            # A size can fail to *compile* rather than mis-compute -- e.g. a long reduction
            # that overruns the shim's buffer-descriptor budget. Report it and keep going.
            msg = sprint(showerror, e)
            note = occursin("buffer descriptors", msg) ? "shim BD limit" : "compile/run failed"
            @printf("%-12s  %5s  %s\n", "$(s)³", "ERR", note)
        end
    end

    # Does routing operands through MemTiles (`L2(...)`) help, and does it scale with cores?
    # Isolate its effect on *tall* GEMMs (large M·K, small N) where A dominates data movement:
    # L2 broadcasts A once instead of re-reading it per core. N = 64/128/256 -> 4/8/16 cores
    # (1/2/4 memtile groups). no-L2 caps at 8 cores (one shim per core), so the 16-core row is
    # L2 only.
    println()
    @printf("%-14s  %5s  %6s  %12s  %12s  %8s\n",
            "M=K x N", "ok", "cores", "no-L2 GF/s", "L2 GF/s", "L2 gain")
    println("-"^66)
    for (s, N) in [(512, 64), (512, 128), (256, 256)]
        M = K = s; m, k, n = 16, 32, 16                  # N/n cores (4, 8, 16)
        ncores = div(N, n)
        try
            a = BFloat16[(i + j) % 7 for i in 1:M, j in 1:K]
            b = BFloat16[(i - 2j) % 5 for i in 1:K, j in 1:N]
            da, db = NPUArray(a), NPUArray(b)
            dc = NPUArray{Tacc}(undef, Tile{Tacc, Tuple{M, N}})
            l2 = time_launch(gemm_launch_l2!, da, db, dc, a, b, M, K, N, m, k, n; trials = 10)
            if ncores <= MAX_CORES
                nol2 = time_launch(gemm_launch_cores!, da, db, dc, a, b, M, K, N, m, k, n; trials = 10)
                @printf("%-14s  %5s  %6d  %12.2f  %12.2f  %7.2fx\n",
                        "$(M)x$(K)x$(N)", (nol2.ok && l2.ok) ? "yes" : "NO", ncores,
                        nol2.gflops, l2.gflops, l2.gflops / nol2.gflops)
            else
                @printf("%-14s  %5s  %6d  %12s  %12.2f  %8s\n",
                        "$(M)x$(K)x$(N)", l2.ok ? "yes" : "NO", ncores, "-", l2.gflops, "L2 only")
            end
        catch e
            @printf("%-14s  %5s  %s\n", "$(M)x$(K)x$(N)", "ERR", sprint(showerror, e))
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
