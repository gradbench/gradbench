# SciLean (Lean 4), built with lean4-nix. It installs the pinned Lean 4.16.0
# toolchain (from lean-toolchain) and turns lake-manifest.json + the Lake build
# graph into normal Nix derivations -- so each dependency (SciLean, mathlib,
# LeanBLAS, ...) is a normal derivation. mathlib's oleans are fetched from the
# Lean community cache (see mathlibCache) rather than elaborated from source.
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

  # mathlib is the slowest dependency to elaborate from source. Instead, fetch
  # its prebuilt .olean/.c from the Lean community's own cache (`lake exe cache
  # get`) -- which serves exactly this rev built with this toolchain -- in a
  # fixed-output derivation. Injecting these lets Lake skip mathlib's
  # elaboration and only compile the .c to .so. The locally-built `cache` exe's
  # artifacts (the only ones referencing the build Lean toolchain) are dropped
  # so this stays a valid, deterministic FOD.
  mathlibCache = pkgs.stdenvNoCC.mkDerivation {
    name = "gradbench-mathlib-cache";
    src = builtins.fetchGit {
      url = "https://github.com/leanprover-community/mathlib4";
      rev = "a6276f4c6097675b1cf5ebd49b1146b735f38c02";
    };
    nativeBuildInputs = [ pkgs.lean.lean-all pkgs.git pkgs.cacert pkgs.curl ];
    dontConfigure = true;
    buildPhase = ''
      export HOME="$TMPDIR"
      lake exe cache get
    '';
    installPhase = ''
      mkdir -p "$out"
      cp -r .lake/build/lib .lake/build/ir "$out/"
      rm -rf "$out/lib/Cache" "$out/ir/Cache" "$out/bin"
    '';
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-2p9UooXQhxFgMF8AjiKm3yeEYdKJRcdeMNAAw/t6xd4=";
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
      # Drop mathlib's prebuilt oleans/.c into .lake/build so Lake skips
      # elaborating mathlib from source and only compiles the .c to .so.
      mathlib = {
        preConfigure = ''
          mkdir -p .lake/build
          cp -r --no-preserve=mode,ownership ${mathlibCache}/. .lake/build/
        '';
        buildPhase = libBuildPhase "mathlib" "Mathlib";
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

  # The compiled `gradbench` executable is fully self-contained: it statically
  # links the Lean runtime, SciLean and mathlib, and at run time opens nothing
  # from the ~35.8 GB Lake build tree (oleans, dereferenced writable dep copies)
  # -- verified by strace; it only needs BLAS and libc (see `ldd`). It does,
  # however, embed the 3.2 GB Lean toolchain path as a dead `.rodata` string
  # (empty rpath, not in DT_NEEDED, never opened). So for the runtime closure we
  # take just the binary and nuke that one unused reference, cutting the closure
  # from ~35.8 GB to a few hundred MB.
  scileanBin = pkgs.runCommand "scilean-bin" {
    nativeBuildInputs = [ pkgs.removeReferencesTo ];
  } ''
    mkdir -p "$out/bin"
    cp ${scilean}/bin/gradbench "$out/bin/gradbench"
    chmod u+w "$out/bin/gradbench"
    remove-references-to -t ${pkgs.lean.lean-all} "$out/bin/gradbench"
  '';
in gblib.mkTool {
  name = "scilean";
  runtimeInputs = [ scileanBin ];
  setup = ''
    export LD_LIBRARY_PATH="${blasLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
  entrypoint = "gradbench";
}
