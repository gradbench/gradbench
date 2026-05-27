# SciLean (Lean 4). WORK IN PROGRESS, not yet wired into the registry.
#
# This builds end to end: elan (from Nixpkgs) fetches the pinned Lean v4.16.0
# (Nixpkgs has only 4.19.0), `lake exe cache get` pulls SciLean + mathlib
# caches, and `lake build` compiles the gradbench binary (the cblas.h include
# and -lblas link are satisfied from Nixpkgs openblas). At run time
# LD_LIBRARY_PATH points at the captured Lean toolchain and BLAS.
#
# The blocker: the Lake build output is NOT bit-reproducible. Pruning the
# installPhase to drop build intermediates (.olean/.c/.o/traces, below) did NOT
# fix it -- three builds gave three different hashes -- so the non-determinism
# is in the Lean-compiled .so/binary themselves, which we can't drop. A pure
# fixed-output derivation therefore can't pin it. To finish, either:
#   - split into a fetch FOD (the deterministic downloads: Lean toolchain, the
#     git deps, and the mathlib .olean cache) + a normal derivation that runs
#     `lake build` OFFLINE against them (a normal derivation needs no output
#     hash, so its non-determinism and store references are fine), or
#   - use an impure derivation (Nix's experimental `impure-derivations`).
# NOTE: the `outputHash` below is therefore stale (kept only so the file
# evaluates); this is not wired into the registry.
{ pkgs, gblib }:

let
  blasPath = pkgs.lib.makeLibraryPath [ pkgs.blas pkgs.openblas ];
  scilean = pkgs.stdenv.mkDerivation {
    name = "gradbench-scilean";
    src = ../../tools/scilean;
    nativeBuildInputs = [
      pkgs.elan
      pkgs.git
      pkgs.cacert
      pkgs.curl
      pkgs.gcc
      pkgs.gnumake
      pkgs.which
    ];
    dontConfigure = true;
    # Lean prebuilts target FHS and use $ORIGIN-relative rpaths; don't let
    # fixupPhase rewrite anything, and keep the build from baking store paths.
    dontPatchShebangs = true;
    buildPhase = ''
      export HOME="$TMPDIR"
      export ELAN_HOME="$TMPDIR/elan"
      export PATH="$ELAN_HOME/bin:$PATH"
      # BLAS for the link step (lakefile uses -lblas via a -L search path) and
      # for LeanBLAS's C sources, which #include <cblas.h>.
      export LIBRARY_PATH="${blasPath}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
      export CPATH="${pkgs.openblas.dev}/include''${CPATH:+:$CPATH}"
      elan toolchain install leanprover/lean4:v4.16.0
      elan default leanprover/lean4:v4.16.0
      lake exe cache get
      lake build buildscilean
      lake build
    '';
    installPhase = ''
      mkdir -p "$out"
      cp -r .lake "$out/.lake"
      # The Lean toolchain provides libleanshared.so etc. at run time.
      cp -r "$ELAN_HOME/toolchains" "$out/toolchains"
      # Keep only runtime artifacts. The gradbench binary is native machine code
      # and links shared libraries; the build intermediates (.olean/.c/.o/traces)
      # are not needed at run time and are the non-reproducible part, so drop
      # them to make this fixed-output derivation deterministic.
      find "$out/.lake" -type f \( -name '*.olean' -o -name '*.ilean' \
        -o -name '*.c' -o -name '*.o' -o -name '*.export' -o -name '*.trace' \
        -o -name '*.hash' -o -name '*.json' -o -name '*.log' \) -delete
      find "$out/.lake" -type d -empty -delete
    '';
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-uq/EJnJ/IydWmba43MSNufT3RHchtI3iukDbG8hBwMY=";
  };
in gblib.mkTool {
  name = "scilean";
  runtimeInputs = [ pkgs.gcc ];
  setup = ''
    leanlib="$(echo ${scilean}/toolchains/*/lib ${scilean}/toolchains/*/lib/lean \
      | tr ' ' ':')"
    export LD_LIBRARY_PATH="$leanlib:${blasPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
  entrypoint = "${scilean}/.lake/build/bin/gradbench";
}
