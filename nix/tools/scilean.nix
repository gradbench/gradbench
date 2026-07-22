# SciLean (Lean 4). The Docker setup ran `lake exe cache get` then
# `lake build buildscilean` in the SciLean source dir; Lake's reachability
# meant only the mathlib modules SciLean actually imports got compiled, so the
# whole build finished in ~22 min. We do the same with two derivations: a
# fixed-output one that does the network step (clones every dep at its
# manifest-pinned revision and fetches the deterministic olean closure from
# the Lean community cache), and a pure compile derivation that consumes its
# output and runs `lake build gradbench`.
#
# The fix for a long-standing build-time regression here: the FOD used to
# run `lake exe cache get` and then strip every non-mathlib dep's
# `.lake/build/` from the result. The strip was the bug -- mathlib's CDN
# ltars cover the full transitive olean closure (mathlib + batteries + aesop
# + Qq + ...), so those oleans were CDN-sourced and bit-deterministic, and
# throwing them away cost the downstream compile a ~108-module mathlib
# rebuild cascade (its stored .trace files referenced batteries oleans by
# depHashes that no longer matched anything on disk). We now keep them.
#
# Why the two-phase dance in buildPhase: `lake exe cache get` interleaves
# (1) building the `cache` exe -- which side-effect-compiles the ~10
# `Cache.*` modules in mathlib, against the BUILD-TIME Lean toolchain, so
# those particular oleans carry Nix store paths that wouldn't match
# CDN-built versions -- and (2) running cache to fetch CDN ltars. We do
# them as two steps with a `rm -rf .lake/build/` between, then run the
# pre-built binary directly so the second step's CDN download repopulates
# the `Cache.*` slot too. Every olean in the FOD output is CDN-sourced.
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

  # FOD: clones every Lake dep at its manifest-pinned revision and fetches the
  # deterministic prebuilt olean closure (mathlib + batteries + aesop + every
  # transitive dep) from the Lean community cache. The downstream compile
  # step consumes the resulting `.lake/` and runs offline.
  scileanLakeCache = pkgs.stdenvNoCC.mkDerivation {
    name = "gradbench-scilean-lake-cache";
    src = ../../tools/scilean;
    nativeBuildInputs = [ pkgs.lean.lean-all pkgs.git pkgs.cacert pkgs.curl ];
    dontConfigure = true;
    dontFixup = true;
    buildPhase = ''
      runHook preBuild
      export HOME="$TMPDIR"

      # Phase 1: build the `cache` exe but don't run it. Lake clones every
      # dep and compiles mathlib's ~10 `Cache.*` modules against the
      # build-time Lean toolchain (Nix store path embedded); those oleans
      # are the artifacts we need the CDN to overwrite.
      lake build cache

      # Snapshot the freshly-built cache binary, then wipe every compiled
      # olean (in `.lake/build/` at the workspace and per-package). After
      # this only sources, .git/, and the saved binary remain.
      cache_bin="$TMPDIR/cache"
      mv .lake/packages/mathlib/.lake/build/bin/cache "$cache_bin"
      rm -rf .lake/build
      for pkg in .lake/packages/*; do
        rm -rf "$pkg/.lake/build"
      done

      # Phase 2: put the binary back and run it directly. Bypassing
      # `lake exe cache get` matters: that command would re-evaluate the
      # `cache` target, see no build dir, and rebuild the `Cache.*` oleans
      # we just threw away. Running the binary directly only triggers the
      # CDN download, which repopulates the entire transitive olean
      # closure (mathlib + batteries + aesop + Qq + ...) from byte-stable
      # URLs, so every compiled artifact this produces is deterministic.
      mkdir -p .lake/packages/mathlib/.lake/build/bin
      mv "$cache_bin" .lake/packages/mathlib/.lake/build/bin/cache
      ./.lake/packages/mathlib/.lake/build/bin/cache get

      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      # Keep the FOD's contents narrow: just what the compile step truly
      # cannot regenerate offline. FODs are fragile -- anything we ship
      # that drifts between machines or wall-clock times (upstream refs,
      # build-env-tagged trace files, locally-built helper exes) becomes
      # a hash-mismatch landmine.
      #
      # Things we KEEP:
      #   * each cloned dep's source tree, at its pinned commit
      #   * every dep's `.lake/build/` (oleans + .c) -- all CDN-sourced
      #     after the two-phase dance, hence bit-deterministic
      #   * a minimal `.git/` per dep so Lake's offline manifest path
      #     recognises the dir as a repo and reads HEAD without fetching
      #
      # Things we DROP:
      #   * the live `.git/` contents (`refs/`, `packed-refs`, the pack
      #     files etc.) -- pack files are non-deterministic across `git
      #     clone` runs, and `packed-refs` captures upstream refs at
      #     clone time, both of which drift between machines/times.
      #     We replace each `.git/` with a hand-built minimal one
      #     (`HEAD` + `config` + empty `objects/`).
      #   * gradbench's and each dep's elaborated `lakefile.olean` and
      #     `lakefile.olean.trace` -- the compile step regenerates these
      #     in seconds, and the `.trace` files include build-env data.
      #   * mathlib's locally-built `cache` exe (`bin/`) -- it's the only
      #     non-CDN artifact left after the two-phase dance, and the
      #     downstream compile doesn't need it.
      #   * proofwidgets' downloaded widget tarball -- we have our own
      #     pinned `fetchurl` and inject it at compile time.
      for pkg in .lake/packages/*; do
        if [ -d "$pkg/.git" ]; then
          rev=$(git -C "$pkg" rev-parse HEAD)
          rm -rf "$pkg/.git"
          # git requires both `objects/` and `refs/` (even if empty) to
          # recognise the dir as a repository at all.
          mkdir -p "$pkg/.git/objects" "$pkg/.git/refs"
          printf '[core]\n\trepositoryformatversion = 0\n' > "$pkg/.git/config"
          printf '%s\n' "$rev" > "$pkg/.git/HEAD"
        fi
      done
      rm -f .lake/lakefile.olean .lake/lakefile.olean.trace
      for pkg in .lake/packages/*; do
        rm -f "$pkg/.lake/lakefile.olean" "$pkg/.lake/lakefile.olean.trace"
      done
      rm -f .lake/packages/proofwidgets/.lake/ProofWidgets4.tar.gz \
            .lake/packages/proofwidgets/.lake/ProofWidgets4.tar.gz.trace
      rm -rf .lake/packages/mathlib/.lake/build/bin
      mkdir -p "$out"
      cp -r .lake "$out/.lake"
      runHook postInstall
    '';
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-UYi1d8v4mSId+I7VcTHWG8uayX1Jm9gkVpoddCgeKOs=";
  };

  # The actual SciLean build. With a populated `.lake/` in place (all deps'
  # oleans matching what their stored .trace files were computed against),
  # Lake skips the dep-rebuild cascade entirely and only compiles the
  # SciLean / gradbench modules. Lake's progress output streams to stderr ->
  # visible in the build log.
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
      # Restore the cached Lake state (cloned deps + CDN-sourced oleans for
      # the full transitive closure) into a writable `.lake/` in this build
      # dir.
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
