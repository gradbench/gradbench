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
    # LeanBLAS dlopens libblas.so.3 during elaboration (build time), so it must
    # also be on LD_LIBRARY_PATH while building.
    LD_LIBRARY_PATH = blasLibPath;
  };

  # ProofWidgets' build bundles editor widget JS with npm, which needs npm +
  # network (unavailable, and unneeded for compiling mathlib or running
  # gradbench). It also publishes that JS prebuilt as a "cloud release"
  # (ProofWidgets4.tar.gz) that Lake would normally fetch. We fetch it and drop
  # it into .lake/build so Lake sees the widgets as up-to-date and skips npm.
  # The tag corresponds to the proofwidgets rev pinned in lake-manifest.json.
  proofwidgetsRelease = pkgs.fetchurl {
    url =
      "https://github.com/leanprover-community/ProofWidgets4/releases/download/v0.0.50/ProofWidgets4.tar.gz";
    hash = "sha256-69d/mQmLSRBD+dIjfJN74Ov+2JoMldUAlk3gNq4Rfmw=";
  };

  # lean4-nix's default buildPhase derives the library name by capitalizing the
  # package's first letter (e.g. leanblas -> Leanblas), which is wrong for these
  # packages, so we provide the correct library name explicitly.
  libBuildPhase = pkg: lib: ''
    runHook preBuild
    lake build ${pkg}
    lake build ${lib}:shared
    lake build ${lib}:static
    runHook postBuild
  '';

  # lean4-nix symlinks each dependency (sources AND .lake artifacts) read-only
  # from the store. SciLean precompiles modules, so a consumer build both writes
  # per-module .so into dependency .lake dirs (EACCES through the symlinks) and
  # may recompile dependency modules -- which Lean rejects when their source is
  # a store symlink ("input file must be contained in root directory"). So we
  # replace each dependency tree with a fully dereferenced, writable real copy,
  # preserving timestamps so Lake doesn't treat artifacts as stale.
  makeDepsWritable = ''
    for pkg in .lake/packages/*; do
      real="$(mktemp -d)"
      cp -rL --preserve=timestamps "$pkg/." "$real/"
      chmod -R u+w "$real"
      chmod u+w "$(dirname "$pkg")"
      find "$pkg" -type d -exec chmod u+w {} + 2>/dev/null || true
      rm -rf "$pkg"
      mv "$real" "$pkg"
    done
  '';

  scilean = lake2nix.mkPackage ({
    name = "gradbench";
    src = ../../tools/scilean;
    depOverride = {
      # LeanBLAS compiles C against <cblas.h>, links -lblas, and dlopens
      # libblas.so.3 during elaboration (hence blasEnv with LD_LIBRARY_PATH).
      leanblas = blasEnv // {
        buildPhase = libBuildPhase "leanblas" "LeanBLAS";
      };
      # SciLean uses LeanBLAS's FFI during elaboration, so it needs BLAS too;
      # and it precompiles modules, so its dependency .lake trees must be
      # writable.
      scilean = blasEnv // {
        postConfigure = makeDepsWritable;
        buildPhase = libBuildPhase "scilean" "SciLean";
      };
      proofwidgets = {
        # Drop the prebuilt widget JS into .lake/build before building, so
        # Lake's widgetJsAll target is up-to-date and never invokes npm.
        preConfigure = ''
          mkdir -p .lake/build
          tar xzf ${proofwidgetsRelease} -C .lake/build
        '';
        buildPhase = libBuildPhase "proofwidgets" "ProofWidgets";
      };
    };
    # The gradbench executable links SciLean's precompiled modules, so its
    # dependency trees must be writable too.
    postConfigure = makeDepsWritable;
    # Build buildscilean first (as the Dockerfile does): it is `import SciLean`,
    # which forces SciLean's modules to be compiled in this package's context
    # before the gradbench executable links them.
    buildPhase = ''
      runHook preBuild
      lake build buildscilean
      lake build gradbench
      runHook postBuild
    '';
  } // blasEnv); # the gradbench executable also links -lblas.
in gblib.mkTool {
  name = "scilean";
  runtimeInputs = [ scilean ];
  setup = ''
    export LD_LIBRARY_PATH="${blasLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
  entrypoint = "gradbench";
}
