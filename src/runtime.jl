# Compiling and running a design on the NPU, with no Python in the loop.
#
# Compilation goes through `aiecc` and Peano (see `compiler/aiecc.jl`); it turns
# the generated MLIR into an xclbin and an NPU instruction stream. Everything
# below that -- opening the device, loading the xclbin, allocating XRT buffers and
# launching the kernel -- is XRT, reached through the small `libironxrt` C shim
# (Yggdrasil/I/ironxrt). The shim exists because XRT's NPU launch path
# (register_xclbin + hw_context + kernel-by-name) lives only in its C++ API, which
# Julia cannot `ccall` directly.

# --- libironxrt bindings -----------------------------------------------------
# Thin wrappers over the shim's C ABI. Each shim entry point is noexcept and
# reports failure by return value; we surface the shim's thread-local last-error.

_xrt_error() = unsafe_string(@ccall libironxrt.ironxrt_last_error()::Cstring)

function _xrt_device_open(index::Integer)
    d = @ccall libironxrt.ironxrt_device_open(index::Cuint)::Ptr{Cvoid}
    d == C_NULL && error("IRON: could not open NPU device $index: $(_xrt_error())")
    return d
end

function _xrt_open(dev::Ptr{Cvoid}, xclbin::AbstractString, kernel::AbstractString)
    ctx = @ccall libironxrt.ironxrt_open(
        dev::Ptr{Cvoid}, xclbin::Cstring, kernel::Cstring
    )::Ptr{Cvoid}
    ctx == C_NULL && error("IRON: could not open NPU design: $(_xrt_error())")
    return ctx
end

_xrt_close(ctx::Ptr{Cvoid}) = @ccall libironxrt.ironxrt_close(ctx::Ptr{Cvoid})::Cvoid

function _xrt_group_id(ctx::Ptr{Cvoid}, argno::Integer)
    g = @ccall libironxrt.ironxrt_group_id(ctx::Ptr{Cvoid}, argno::Cint)::Cint
    g < 0 && error("IRON: could not get group id for arg $argno: $(_xrt_error())")
    return g
end

function _xrt_bo_alloc(dev::Ptr{Cvoid}, nbytes::Integer, group_id::Integer, cacheable::Bool)
    bo = @ccall libironxrt.ironxrt_bo_alloc(
        dev::Ptr{Cvoid}, nbytes::Csize_t, group_id::Cint, cacheable::Cint
    )::Ptr{Cvoid}
    bo == C_NULL && error("IRON: could not allocate a $nbytes-byte NPU buffer: $(_xrt_error())")
    return bo
end

function _xrt_bo_map(bo::Ptr{Cvoid})
    p = @ccall libironxrt.ironxrt_bo_map(bo::Ptr{Cvoid})::Ptr{Cvoid}
    p == C_NULL && error("IRON: could not map NPU buffer: $(_xrt_error())")
    return p
end

# A null buffer -- a placeholder or an already-freed one -- has nothing to sync, so
# these are no-ops on `C_NULL` rather than a call into the shim, matching how
# `_free_bo!` guards a null handle. That also lets an `NPUArray` built over a plain
# host array (no device) be read and shown without touching XRT.
function _xrt_bo_sync_to_device(bo::Ptr{Cvoid})
    bo == C_NULL && return nothing
    r = @ccall libironxrt.ironxrt_bo_sync_to_device(bo::Ptr{Cvoid})::Cint
    r == 0 || error("IRON: sync to device failed: $(_xrt_error())")
    return nothing
end

function _xrt_bo_sync_from_device(bo::Ptr{Cvoid})
    bo == C_NULL && return nothing
    r = @ccall libironxrt.ironxrt_bo_sync_from_device(bo::Ptr{Cvoid})::Cint
    r == 0 || error("IRON: sync from device failed: $(_xrt_error())")
    return nothing
end

_xrt_bo_free(bo::Ptr{Cvoid}) = @ccall libironxrt.ironxrt_bo_free(bo::Ptr{Cvoid})::Cvoid

function _xrt_run(ctx::Ptr{Cvoid}, instr::Ptr{Cvoid}, ninstr::Integer, args::Vector{Ptr{Cvoid}})
    r = GC.@preserve args @ccall libironxrt.ironxrt_run(
        ctx::Ptr{Cvoid}, instr::Ptr{Cvoid}, ninstr::Cuint,
        pointer(args)::Ptr{Ptr{Cvoid}}, length(args)::Cuint
    )::Cint
    r == 0 || error("IRON: NPU launch failed: $(_xrt_error())")
    return nothing
end

# --- process-wide device -----------------------------------------------------
# One device handle, opened on first use and shared by every buffer and launch --
# the way MLIR-AIE's Python runtime keeps a single `pyxrt.device(0)`. It lives for
# the process; there is nothing to release it against.

const _DEVICE = Ref{Ptr{Cvoid}}(C_NULL)

function _device!()
    if _DEVICE[] == C_NULL
        _DEVICE[] = _xrt_device_open(0)
    end
    return _DEVICE[]
end

# --- NPUArray allocation (used by array.jl) ----------------------------------

# Allocate an NPU-resident, host-mapped buffer of shape `dims`, its contents left
# uninitialized. Data buffers are host-only in bank 0, the default the Python
# XRTTensor uses; the caller fills and syncs it. The mapping is wrapped as a
# column-major array over the buffer, so it indexes like any Julia array.
function _npu_empty(::Type{T}, dims::Dims{N}) where {T, N}
    dev = _device!()
    bo = _xrt_bo_alloc(dev, prod(dims) * sizeof(T), 0, false)
    data = unsafe_wrap(Array, Ptr{T}(_xrt_bo_map(bo)), dims; own = false)
    a = NPUArray{T, N}(bo, data)
    finalizer(_free_bo!, a)
    return a
end

function _free_bo!(a::NPUArray)
    a.bo == C_NULL || _xrt_bo_free(a.bo)
    a.bo = C_NULL
    return nothing
end

# --- compiled program --------------------------------------------------------

"""
    CompiledProgram

A [`Program`](@ref) lowered to an xclbin plus its NPU instruction stream, ready
to run. The XRT launch context and the instruction buffer are opened lazily on
the first [`run!`](@ref) and reused thereafter; they are released when the
`CompiledProgram` is finalized.
"""
mutable struct CompiledProgram
    program::Program
    xclbin::String
    insts::Vector{UInt32}
    ctx::Ptr{Cvoid}       # xclbin+hw_context+kernel; C_NULL until first run
    instr_bo::Ptr{Cvoid}  # cached instruction buffer; C_NULL until first run
end

function _release!(c::CompiledProgram)
    c.instr_bo == C_NULL || _xrt_bo_free(c.instr_bo)
    c.ctx == C_NULL || _xrt_close(c.ctx)
    c.instr_bo = C_NULL
    c.ctx = C_NULL
    return nothing
end

# Load the raw little-endian UInt32 instruction stream aiecc emits.
function _load_insts(path::AbstractString)
    bytes = read(path)
    length(bytes) % sizeof(UInt32) == 0 ||
        error("IRON: instruction stream $path is not a whole number of 32-bit words")
    return collect(reinterpret(UInt32, bytes))
end

"""
    compile(program; path=nothing, workdir=mktempdir(), flags=String[], verbose=false)
        -> CompiledProgram

Generate `program`'s MLIR, compile it to an NPU xclbin + instruction stream with
`aiecc`/Peano, and wrap the result. `path`, if given, is where the `.mlir` is
written; otherwise it goes under `workdir`. `flags` are passed through to `aiecc`
(e.g. `["--alloc-scheme=basic-sequential"]`). `peano` overrides the Peano/llvm-aie
install used for per-core codegen and linking (see [`aiecc_compile`](@ref)).
"""
function compile(
    p::Program;
    path::Union{Nothing, AbstractString} = nothing,
    workdir::AbstractString = mktempdir(),
    peano::AbstractString = AIE_LLVM_Toolchain_jll.artifact_dir,
    flags::AbstractVector{<:AbstractString} = String[],
    verbose::Bool = false,
)
    isdir(workdir) || mkpath(workdir)
    mlir = generate_mlir(p)
    mlir_file = path === nothing ? joinpath(workdir, "aie.mlir") : String(path)
    write(mlir_file, mlir)
    xclbin, insts = aiecc_compile(mlir_file; workdir, peano, flags, verbose)
    return CompiledProgram(p, xclbin, _load_insts(insts), C_NULL, C_NULL)
end

# Open the launch context on first use and cache it, registering the finalizer
# that will release it (and the instruction buffer).
function _context!(c::CompiledProgram)
    if c.ctx == C_NULL
        c.ctx = _xrt_open(_device!(), c.xclbin, KERNEL_NAME)
        finalizer(_release!, c)
    end
    return c.ctx
end

# Upload the instruction stream once, into a cacheable buffer in the bank XRT
# assigns to kernel argument 1 (the fixed NPU layout: opcode, instr, count, ...).
function _instr_bo!(c::CompiledProgram, ctx::Ptr{Cvoid})
    if c.instr_bo == C_NULL
        gid = _xrt_group_id(ctx, 1)
        bo = _xrt_bo_alloc(_device!(), sizeof(c.insts), gid, true)
        n = sizeof(c.insts)
        GC.@preserve c unsafe_copyto!(Ptr{UInt8}(_xrt_bo_map(bo)), Ptr{UInt8}(pointer(c.insts)), n)
        _xrt_bo_sync_to_device(bo)
        c.instr_bo = bo
    end
    return c.instr_bo
end

"""
    run!(compiled, arrays...) -> nothing

Run the design on the NPU. `arrays` are the NPU-resident [`NPUArray`](@ref)s -- one
per runtime sequence argument, in order. Their buffers are used in place: inputs
are flushed to the device before the launch, outputs read back after; copy them to
the host with `Array`.

The first call opens the device and loads the xclbin; later calls reuse them.
"""
function run!(c::CompiledProgram, arrays::NPUArray...)
    length(arrays) == length(c.program.argtypes) || error(
        "IRON: design takes $(length(c.program.argtypes)) buffers, got $(length(arrays))"
    )
    ctx = _context!(c)
    instr = _instr_bo!(c, ctx)

    # The design's resident buffers, in argument order. Flush each to the device
    # (inputs may have been written since the last sync), launch, then pull each
    # back so a host read sees the design's output.
    bos = Ptr{Cvoid}[buffer(a) for a in arrays]
    for bo in bos
        _xrt_bo_sync_to_device(bo)
    end
    _xrt_run(ctx, instr, length(c.insts), bos)
    for bo in bos
        _xrt_bo_sync_from_device(bo)
    end
    return nothing
end
