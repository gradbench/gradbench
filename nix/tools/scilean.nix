# SciLean (Lean 4). WORK IN PROGRESS, not yet wired into the registry.
#
# This builds end to end: elan (from Nixpkgs) fetches the pinned Lean v4.16.0
# (Nixpkgs has only 4.19.0), `lake exe cache get` pulls SciLean + mathlib
# caches, and `lake build` compiles the gradbench binary (the cblas.h include
# and -lblas link are satisfied from Nixpkgs openblas). At run time
# LD_LIBRARY_PATH points at the captured Lean toolchain and BLAS.
#
# The blocker: the Lake/mathlib build output is NOT bit-reproducible (two builds
# produced two different hashes), so it can't be a fixed-output derivation as
# written. To finish, either:
#   - prune installPhase to only the runtime artifacts (the gradbench binary and
#     the .so libraries it needs) and confirm those are deterministic, or
#   - use an impure derivation (Nix's experimental `impure-derivations`).
# NOTE: the `outputHash` below is therefore stale.
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
    '';
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-4U/pNL8hcgRg1Dn7X2AtYkJLQ56aabPcNdlq0RqMkVA=";
  };
in gblib.mkTool {
  name = "scilean";
  runtimeInputs = [ pkgs.gcc ];
  setup = ''
    leanlib="$(echo ${scilean}/toolchains/*/lib ${scilean}/toolchains/*/lib/lean)"
    export LD_LIBRARY_PATH="$leanlib:${blasPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
  entrypoint = "${scilean}/.lake/build/bin/gradbench";
}
