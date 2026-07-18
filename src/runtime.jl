# Compiling and running a design on the NPU, with no Python in the loop.
#
# Compilation goes through `aiecc` and Peano (see `compiler/aiecc.jl`); it turns
# the generated MLIR into an xclbin and an NPU instruction stream. Everything
# below that -- opening the device, loading the xclbin, allocating XRT buffers and
# launching the kernel -- is XRT, through XRT.jl's CxxWrap wrapper (`XRT.XRTWrap`).
# The NPU launch path is register_xclbin + hw_context + kernel-by-name; the buffer
# objects and the launch context are CxxWrap objects, released by the GC.

# --- XRT.jl bindings ---------------------------------------------------------
# `Bo` is a resident buffer object; `nothing` is a buffer with no device backing
# (an NPUArray built over a plain host array), which has nothing to sync or map.

const Bo = XRT.XRTWrap.BO
const _TO_DEVICE = XRT.XRTWrap.BOSyncDirection.TO_DEVICE
const _FROM_DEVICE = XRT.XRTWrap.BOSyncDirection.FROM_DEVICE

# A buffer of `nbytes` on `dev` in bank `group`. Data buffers are host-only, the
# same default the Python XRTTensor uses; the instruction stream is cacheable.
function _bo_alloc(dev, nbytes::Integer, group, cacheable::Bool)
    flags = cacheable ? XRT.XRTWrap.BOFlags.CACHEABLE : XRT.XRTWrap.BOFlags.HOST_ONLY
    return Bo(dev, Csize_t(nbytes), flags, group)
end

_bo_map(bo::Bo) = XRT.XRTWrap.map(bo)

_bo_sync_to_device(bo::Bo) = (XRT.XRTWrap.sync!(bo, _TO_DEVICE); nothing)
_bo_sync_to_device(::Nothing) = nothing
_bo_sync_from_device(bo::Bo) = (XRT.XRTWrap.sync!(bo, _FROM_DEVICE); nothing)
_bo_sync_from_device(::Nothing) = nothing

# Open the design: register the xclbin, open a hardware context on it and make the
# kernel by name. The three objects are returned together because each must stay
# alive for the kernel to be usable.
function _open_design(dev, xclbin_path::AbstractString, kernel_name::AbstractString)
    xclbin = XRT.XRTWrap.Xclbin(String(xclbin_path))
    uuid = XRT.XRTWrap.register_xclbin(dev, xclbin)
    hwctx = XRT.XRTWrap.HwContext(dev, uuid)
    kernel = XRT.XRTWrap.Kernel(hwctx, String(kernel_name))
    return xclbin, hwctx, kernel
end

# The bank XRT assigns to a kernel argument.
_group_id(kernel, argno::Integer) = XRT.XRTWrap.group_id(kernel, Cint(argno))

# Launch: the fixed NPU argument layout is opcode (3), instruction BO, instruction
# word count, then the data BOs in order. Scalars go as 64-bit; XRT copies only as
# many bytes as the kernel's argument declares.
function _run(kernel, instr::Bo, ninstr::Integer, args::Vector{Bo})
    run = XRT.Run(kernel)
    XRT.set_arg!(run, 0, UInt64(3))
    XRT.set_arg!(run, 1, instr)
    XRT.set_arg!(run, 2, UInt64(ninstr))
    for (i, bo) in enumerate(args)
        XRT.set_arg!(run, 2 + i, bo)
    end
    XRT.start(run)
    XRT.wait(run)
    return nothing
end

# --- process-wide device -----------------------------------------------------
# One device handle, opened on first use and shared by every buffer and launch --
# the way MLIR-AIE's Python runtime keeps a single `pyxrt.device(0)`. It lives for
# the process; there is nothing to release it against.

const _DEVICE = Ref{Union{XRT.XRTWrap.Device, Nothing}}(nothing)

function _device!()
    if _DEVICE[] === nothing
        _DEVICE[] = XRT.XRTWrap.Device(0)
    end
    return _DEVICE[]
end

# --- NPUArray allocation (used by array.jl) ----------------------------------

# Allocate an NPU-resident, host-mapped buffer of shape `dims`, its contents left
# uninitialized. Data buffers are host-only in bank 0, the default the Python
# XRTTensor uses; the caller fills and syncs it. The mapping is wrapped as a
# column-major array over the buffer, so it indexes like any Julia array. The
# buffer object is owned by the NPUArray and freed with it by the GC.
function _npu_empty(::Type{T}, dims::Dims{N}) where {T, N}
    dev = _device!()
    bo = _bo_alloc(dev, prod(dims) * sizeof(T), 0, false)
    data = unsafe_wrap(Array, Ptr{T}(_bo_map(bo)), dims; own = false)
    return NPUArray{T, N}(bo, data)
end

# --- compiled program --------------------------------------------------------

"""
    CompiledProgram

A [`Program`](@ref) lowered to an xclbin plus its NPU instruction stream, ready
to run. The XRT launch context and the instruction buffer are opened lazily on
the first [`run!`](@ref) and reused thereafter; they are released with the
`CompiledProgram` by the garbage collector.
"""
mutable struct CompiledProgram
    nargs::Int  # number of runtime-sequence buffers the design takes
    xclbin::String
    insts::Vector{UInt32}
    # The launch context and instruction buffer, opened lazily on the first run! and
    # reused. `nothing` until then; the XRT objects are released with the CompiledProgram
    # by the GC. `xclbin_obj` and `hwctx` are held only to keep the kernel usable.
    xclbin_obj::Union{XRT.XRTWrap.Xclbin, Nothing}
    hwctx::Union{XRT.XRTWrap.HwContext, Nothing}
    kernel::Union{XRT.XRTWrap.Kernel, Nothing}
    instr_bo::Union{Bo, Nothing}
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
    compile(mlir::AbstractString, nargs; kwargs...) -> CompiledProgram

Compile a design to an NPU xclbin + instruction stream with `aiecc`/Peano, and wrap
the result. The first form generates `program`'s MLIR; the second takes MLIR text
directly together with `nargs`, the number of runtime-sequence buffers it takes.
`path`, if given, is where the `.mlir` is written; otherwise it goes under `workdir`.
`flags` are passed through to `aiecc` (e.g. `["--alloc-scheme=basic-sequential"]`).
`peano` overrides the Peano/llvm-aie install used for per-core codegen and linking
(see [`aiecc_compile`](@ref)).
"""
function compile(
    mlir::AbstractString, nargs::Integer;
    path::Union{Nothing, AbstractString} = nothing,
    workdir::AbstractString = mktempdir(),
    peano::AbstractString = AIE_LLVM_Toolchain_jll.artifact_dir,
    flags::AbstractVector{<:AbstractString} = String[],
    verbose::Bool = false,
)
    isdir(workdir) || mkpath(workdir)
    mlir_file = path === nothing ? joinpath(workdir, "aie.mlir") : String(path)
    write(mlir_file, mlir)
    xclbin, insts = aiecc_compile(mlir_file; workdir, peano, flags, verbose)
    return CompiledProgram(Int(nargs), xclbin, _load_insts(insts), nothing, nothing, nothing, nothing)
end

compile(p::Program; kwargs...) = compile(generate_mlir(p), length(p.argtypes); kwargs...)

# Open the kernel on first use and cache it. The XRT objects are released with the
# CompiledProgram by the GC, so there is no finalizer to register.
function _kernel!(c::CompiledProgram)
    if c.kernel === nothing
        c.xclbin_obj, c.hwctx, c.kernel = _open_design(_device!(), c.xclbin, KERNEL_NAME)
    end
    return c.kernel
end

# Upload the instruction stream once, into a cacheable buffer in the bank XRT
# assigns to kernel argument 1 (the fixed NPU layout: opcode, instr, count, ...).
function _instr_bo!(c::CompiledProgram, kernel)
    if c.instr_bo === nothing
        gid = _group_id(kernel, 1)
        bo = _bo_alloc(_device!(), sizeof(c.insts), gid, true)
        n = sizeof(c.insts)
        GC.@preserve c unsafe_copyto!(Ptr{UInt8}(_bo_map(bo)), Ptr{UInt8}(pointer(c.insts)), n)
        _bo_sync_to_device(bo)
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
    length(arrays) == c.nargs || error(
        "IRON: design takes $(c.nargs) buffers, got $(length(arrays))"
    )
    kernel = _kernel!(c)
    instr = _instr_bo!(c, kernel)

    # The design's resident buffers, in argument order. Flush each to the device
    # (inputs may have been written since the last sync), launch, then pull each
    # back so a host read sees the design's output.
    bos = Bo[buffer(a) for a in arrays]
    for bo in bos
        _bo_sync_to_device(bo)
    end
    _run(kernel, instr, length(c.insts), bos)
    for bo in bos
        _bo_sync_from_device(bo)
    end
    return nothing
end
