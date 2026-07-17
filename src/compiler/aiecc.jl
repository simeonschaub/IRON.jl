# Compiling generated MLIR to an NPU xclbin + instruction stream.
#
# This is the Python-free replacement for the compile half of `aie.iron.jit`.
# The whole toolchain comes from JLL artifacts, so nothing outside Julia's
# package depot is needed -- no MLIR-AIE checkout, no ironenv:
#
#   * `mlir_aie_jll` ships the native `aiecc` compile driver (plus `aie-opt` /
#     `aie-translate`). `aiecc` orchestrates the full lowering, links bootgen for
#     PDI generation, and packages the xclbin -- everything the Python `aiecc.py`
#     did, in one executable.
#   * `Peano_jll` ships the AIE LLVM fork (`clang`/`lld`/`llc`/`opt`) that `aiecc`
#     shells out to for per-core code generation. It is handed to `aiecc` via
#     `--peano=<dir>` rather than the chess/xchesscc path (which needs the
#     proprietary Vitis toolchain we do not have).
#
# The flag set mirrors MLIR-AIE's own peano-path invocation in
# `python/utils/compile/utils.py:compile_mlir_module`, so a design gets the same
# lowering it would through the Python stack.

"""
    aiecc_compile(mlir_file; workdir, xclbin, insts, flags=String[], verbose=false)
        -> (xclbin, insts)

Compile the AIE MLIR in `mlir_file` to an NPU `xclbin` and its NPU instruction
stream (`insts`, a little-endian `UInt32` blob), returning the two paths.

`aiecc` emits the kernel under the default name `MLIR_AIE`, which is what the
runtime opens. Intermediate build products land in `workdir`.

`peano` is the Peano/llvm-aie install `aiecc` shells out to for per-core code
generation and linking; it defaults to `Peano_jll`. Override it to point at
another llvm-aie (e.g. one whose AIE bare-metal runtimes -- `libclang_rt.builtins.a`,
`crt0.o`/`crt1.o`, `libc.a`/`libm.a` -- are present, which `Peano_jll` does not
yet ship).

`flags` are passed through to `aiecc` verbatim; the one worth knowing is
`--alloc-scheme=basic-sequential`, which forces sequential buffer allocation for
designs whose bank-aware allocation silently overlaps buffers (see the README).
"""
function aiecc_compile(
    mlir_file::AbstractString;
    workdir::AbstractString = mktempdir(),
    xclbin::AbstractString = joinpath(workdir, "design.xclbin"),
    insts::AbstractString = joinpath(workdir, "insts.bin"),
    peano::AbstractString = Peano_jll.artifact_dir,
    flags::AbstractVector{<:AbstractString} = String[],
    verbose::Bool = false,
)
    isdir(workdir) || mkpath(workdir)
    args = String[
        "--no-compile-host",        # we only want the device artifacts
        "--no-xchesscc",            # use Peano, not the Vitis chess front-end
        "--no-xbridge",             # ... and Peano's lld for linking too
        "--peano=$peano",
        "--aie-generate-npu-insts", "--npu-insts-name=$insts",
        "--aie-generate-xclbin", "--xclbin-name=$xclbin",
        "--tmpdir=$workdir",
    ]
    verbose && push!(args, "--verbose")
    append!(args, flags)
    # `aiecc()` yields a Cmd carrying the JLL's library environment; the Peano
    # tools it spawns are found through `--peano` above.
    run(`$(aiecc()) $mlir_file $args`)
    return (xclbin, insts)
end
