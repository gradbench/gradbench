# SciLean (Lean 4), built with lean4-nix. It installs the pinned Lean 4.16.0
# toolchain (from lean-toolchain) and turns lake-manifest.json + the Lake build
# graph into normal Nix derivations -- so each dependency (SciLean, mathlib,
# LeanBLAS, ...) is a normal derivation built from source. mathlib is not in
# Nixpkgs, so it is compiled from source here (large, but cacheable).
#
# LeanBLAS compiles C against <cblas.h> and the final binary links -lblas, so we
# inject BLAS (from Nixpkgs) into the leanblas dependency build and the gradbench
# build, and onto LD_LIBRARY_PATH at run time.
{ lean4-nix, system, gblib }:

let
  pkgs = import lean4-nix.inputs.nixpkgs {
    inherit system;
    overlays =
      [ (lean4-nix.readToolchainFile ../../tools/scilean/lean-toolchain) ];
  };
  lake2nix = pkgs.callPackage lean4-nix.lake { };

  blasLibPath = pkgs.lib.makeLibraryPath [ pkgs.blas pkgs.openblas ];
  blasEnv = {
    CPATH = "${pkgs.openblas.dev}/include";
    LIBRARY_PATH = blasLibPath;
  };

  scilean = lake2nix.mkPackage ({
    name = "gradbench";
    src = ../../tools/scilean;
    # LeanBLAS needs <cblas.h> to compile its C sources.
    depOverride = { leanblas = blasEnv; };
  } // blasEnv); # ... and the gradbench executable links -lblas.
in gblib.mkTool {
  name = "scilean";
  runtimeInputs = [ scilean ];
  setup = ''
    export LD_LIBRARY_PATH="${blasLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
  entrypoint = "gradbench";
}
