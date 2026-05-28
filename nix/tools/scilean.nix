# SciLean (Lean 4). The Docker setup ran `lake exe cache get` then
# `lake build buildscilean` in the SciLean source dir; Lake's reachability
# meant only the mathlib modules SciLean actually imports got compiled, so the
# whole build finished in ~22 min. We do the same with two derivations: a
# fixed-output one that runs the network step (`lake exe cache get`, which
# clones every dep at its manifest-pinned revision and fetches mathlib's
# prebuilt oleans + .c from the Lean community cache), and a pure compile
# derivation that consumes its output and runs `lake build gradbench`.
{ lean4-nix, system, gblib }:

let
  pkgs = import lean4-nix.inputs.nixpkgs {
    inherit system;
    overlays =
      [ (lean4-nix.readToolchainFile ../../tools/scilean/lean-toolchain) ];
  };

  blasLibPath = pkgs.lib.makeLibraryPath [ pkgs.blas pkgs.openblas ];
  # LeanBLAS compiles C against <cblas.h>, links -lblas, and dlopens
  # libblas.so.3 during elaboration, so BLAS must be visible to both the
  # compiler and the linker (and on LD_LIBRARY_PATH at build time).
  blasEnv = {
    CPATH = "${pkgs.openblas.dev}/include";
    LIBRARY_PATH = blasLibPath;
    LD_LIBRARY_PATH = blasLibPath;
  };

  # ProofWidgets' build bundles editor widget JS with npm (needs network +
  # node), but it also publishes that JS as a "cloud release" tarball Lake
  # would fetch at build time. We pre-fetch the tarball and inject it during
  # compile so Lake's `widgetJsAll` target is up to date and skips npm.
  # The tag matches the proofwidgets rev pinned in lake-manifest.json.
  proofwidgetsRelease = pkgs.fetchurl {
    url =
      "https://github.com/leanprover-community/ProofWidgets4/releases/download/v0.0.50/ProofWidgets4.tar.gz";
    hash = "sha256-69d/mQmLSRBD+dIjfJN74Ov+2JoMldUAlk3gNq4Rfmw=";
  };

  # FOD that does what `RUN lake exe cache get` does in the Dockerfile: it
  # clones every Lake dep at its manifest-pinned revision and fetches the
  # prebuilt mathlib `.olean`/`.c` from the Lean community cache. We capture
  # the resulting `.lake/` so the downstream compile step is pure (no
  # network, no clones).
  scileanLakeCache = pkgs.stdenvNoCC.mkDerivation {
    name = "gradbench-scilean-lake-cache";
    src = ../../tools/scilean;
    nativeBuildInputs = [ pkgs.lean.lean-all pkgs.git pkgs.cacert pkgs.curl ];
    dontConfigure = true;
    dontFixup = true;
    buildPhase = ''
      runHook preBuild
      export HOME="$TMPDIR"
      lake exe cache get
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      # In each cloned dep's `.git/`, keep only the small, deterministic
      # metadata Lake actually needs (`config`, `HEAD`, `refs/`,
      # `packed-refs`) and an empty `objects/` -- git refuses to recognise
      # the dir as a repo without `objects/`. Pack files, logs, index, etc.
      # are non-deterministic across `git clone` runs, so we drop them.
      # Lake's manifest path is offline-aware: when the dir exists and
      # `getHeadRevision?` already matches the pinned rev, it skips
      # `git fetch` entirely (see Lake's Materialize.lean), so this minimal
      # `.git/` is enough for the compile step to proceed without network.
      for pkg in .lake/packages/*; do
        if [ -d "$pkg/.git" ]; then
          find "$pkg/.git" -mindepth 1 -maxdepth 1 \
            ! -name config ! -name HEAD ! -name refs ! -name packed-refs \
            -exec rm -rf {} +
          mkdir -p "$pkg/.git/objects"
        fi
      done
      # Drop the locally-built `cache` exe artifacts: they reference the
      # build-time Lean toolchain and we only needed them to fetch the
      # olean cache.
      rm -rf .lake/packages/mathlib/.lake/build/bin
      rm -rf .lake/packages/mathlib/.lake/build/lib/Cache
      rm -rf .lake/packages/mathlib/.lake/build/ir/Cache
      mkdir -p "$out"
      cp -r .lake "$out/.lake"
      runHook postInstall
    '';
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-yq57kvmwSEbRxB4M+5Zb+o/cXOw+/IV7cI/BKI5IG0U=";
  };

  # The actual SciLean build. With a populated `.lake/` in place, Lake
  # compiles only the modules reachable from `gradbench` (same subset as
  # Docker's `lake build buildscilean`), so the build takes ~20-30 min
  # instead of compiling the entire `Mathlib:shared` library from scratch.
  # Lake's progress output streams to stderr -> visible in the build log.
  scilean = pkgs.stdenv.mkDerivation (blasEnv // {
    name = "gradbench-scilean";
    src = ../../tools/scilean;
    # Lake shells out to `git` (e.g. `git rev-parse HEAD`) to validate that
    # the cached deps in `.lake/packages/*` are at the manifest's pinned
    # revisions; without git on PATH it falls into the update / "URL has
    # changed" path and tries to clone, which fails offline.
    nativeBuildInputs = [ pkgs.lean.lean-all pkgs.git ];
    buildInputs = [ pkgs.blas pkgs.openblas ];
    # LeanBLAS calls `cblas_daxpby`, which openblas *exports* but its
    # `cblas.h` does not *declare*. Under nixpkgs gcc 14 that becomes an
    # error (`-Werror=implicit-function-declaration` is the default); turn
    # it back into a warning so the link-time symbol resolves normally.
    # The previous lean4-nix-based build sidestepped this by not going
    # through the nixpkgs cc wrapper.
    env.NIX_CFLAGS_COMPILE = "-Wno-error=implicit-function-declaration";
    dontConfigure = true;
    buildPhase = ''
      runHook preBuild
      # Restore the cached Lake state (cloned deps + mathlib oleans) into a
      # writable `.lake/` in this build dir.
      cp -r ${scileanLakeCache}/.lake .lake
      chmod -R u+w .lake
      # Inject ProofWidgets' prebuilt widget JS so its build sees the
      # widgets as up-to-date and never invokes npm.
      mkdir -p .lake/packages/proofwidgets/.lake/build
      tar xzf ${proofwidgetsRelease} -C .lake/packages/proofwidgets/.lake/build
      # Build buildscilean first (as the Dockerfile does): it is `import
      # SciLean`, which forces SciLean's modules to compile in this
      # package's context before the gradbench executable links them.
      lake build buildscilean
      lake build gradbench
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p "$out/bin"
      cp .lake/build/bin/gradbench "$out/bin/"
      runHook postInstall
    '';
  });

  # The compiled `gradbench` executable is fully self-contained: it statically
  # links the Lean runtime, SciLean and mathlib, and at run time opens nothing
  # from the build tree (verified by strace; empty rpath, not in DT_NEEDED).
  # It does embed the 3.2 GB Lean toolchain path as a dead `.rodata` string;
  # we copy out just the binary and nuke that one unused reference, so the
  # runtime closure is ~400 MB rather than the build tree.
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
