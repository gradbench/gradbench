# Helpers that capture the patterns previously encoded in the per-eval and
# per-tool Dockerfiles. Each eval/tool becomes a native runner (a wrapper
# script that bakes in dependencies + entrypoint) plus an OCI image built from
# that same runner's closure via `dockerTools`.
{ pkgs, src }:

let inherit (pkgs) lib;
in rec {
  # The nlohmann/json single header, exposed as an include directory so it can
  # be put on CPATH (the C++ sources do `#include "json.hpp"`). This replaces
  # the `wget` in cpp/Makefile.
  jsonInclude = pkgs.runCommand "gradbench-json-include" {
    header = pkgs.fetchurl {
      url =
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp";
      hash = "sha256-m+pMgGbvShwgayvlo2MC+JJvf9xgh69dILQX0M8QPqY=";
    };
  } ''
    mkdir -p "$out/include"
    cp "$header" "$out/include/json.hpp"
  '';

  # A native runner. Equivalent to a Dockerfile's ENTRYPOINT plus the
  # environment it ran in. `setup` is raw shell (e.g. `export` lines) executed
  # before the entrypoint, and `entrypoint` is the command to exec.
  #
  # Each run gets a fresh writable workdir initialised from `embeddedSource`
  # (the gradbench source baked into the runner's closure), with
  # GRADBENCH_SOURCE_ROOT pointing at it. The user's working tree is never
  # read or written to, so an eval/tool can't observe artifacts left behind
  # by a different (eval, tool) pair's run. Compile-on-demand binaries land
  # in this tmpdir and are discarded on exit -- which is also what we want
  # for benchmark timing (no stale `bin/` between runs).
  mkRunner = { name, runtimeInputs ? [ ], setup ? "", entrypoint }:
    pkgs.writeShellApplication {
      inherit name;
      runtimeInputs = [ pkgs.coreutils ] ++ runtimeInputs;
      text = ''
        # Scrub the inherited environment so the caller's shell (in
        # particular our `nix develop` dev shell, which puts every
        # buildInput's lib on LD_LIBRARY_PATH) can't paper over runtime
        # deps that this runner's closure forgot to declare. Re-exec under
        # `env -i` with a short allowlist; the writeShellApplication
        # preamble that runs after the re-exec then rebuilds PATH from
        # `runtimeInputs` alone (no parent PATH leakage). Matches what CI
        # sees on a fresh Ubuntu runner without a dev shell loaded.
        # XDG_RUNTIME_DIR and DBUS_SESSION_BUS_ADDRESS are preserved so
        # the cgroup-scope wrapper below can reach the user's systemd
        # session manager.
        if [ -z "''${GRADBENCH_RUNNER_CLEAN:-}" ]; then
          exec env -i \
            HOME="''${HOME:-/tmp}" \
            USER="''${USER:-nobody}" \
            TERM="''${TERM:-dumb}" \
            LANG="''${LANG:-C.UTF-8}" \
            TMPDIR="''${TMPDIR:-/tmp}" \
            XDG_RUNTIME_DIR="''${XDG_RUNTIME_DIR:-}" \
            DBUS_SESSION_BUS_ADDRESS="''${DBUS_SESSION_BUS_ADDRESS:-}" \
            GRADBENCH_CONTAIN_OOM="''${GRADBENCH_CONTAIN_OOM:-}" \
            GRADBENCH_RUNNER_CLEAN=1 \
            "$0" "$@"
        fi

        root="$(mktemp -d -t gradbench-${name}.XXXXXX)"
        trap 'rm -rf "$root"' EXIT
        # Preserve mode so the pre-compiled `tools/manual/bin/*` baselines
        # keep their executable bit; then add user-write on top so
        # compile-on-demand can overwrite anything it owns.
        cp -r --no-preserve=ownership ${embeddedSource}/. "$root/"
        chmod -R u+w "$root"
        export GRADBENCH_SOURCE_ROOT="$root"
        ${setup}
        cd "$root"

        # Run the entrypoint inside its own transient cgroup scope so an
        # OOM kill in the eval/tool stays contained: the kernel reaps
        # the heaviest process in the scope, `OOMPolicy=continue` keeps
        # systemd from SIGTERM'ing the rest of the scope after, and the
        # in-process wrapper (cpp.py around `tools/<tool>/bin/<m>`,
        # Julia/Lean runtimes, etc.) survives long enough to emit the
        # `{success: false, status: -9, ...}` response the harness needs
        # to log `outcome failure`. Mirrors the per-container memory
        # boundary the old Docker setup had; without it the OOM would
        # cascade up to the GHA runner agent and exit 143 mid-protocol.
        #
        # Two ways to enter a transient scope:
        #
        #   1. `systemd-run --user --scope`: the friendly path. Needs a
        #      user systemd session (XDG_RUNTIME_DIR + a DBus socket).
        #      Works locally under `nix develop` but not on GHA, where
        #      the runner user has no login session and no user manager.
        #
        #   2. `sudo systemd-run --scope --uid/--gid`: system-level
        #      transient scope, no user manager required. Drops back to
        #      the caller's uid/gid via systemd's own setuid path. Works
        #      on GHA (the runner has passwordless sudo) and on any
        #      Linux host where `sudo -n` is preconfigured. Opt-in via
        #      the gradbench CLI's `--contain-oom` flag (sets
        #      `GRADBENCH_CONTAIN_OOM=1` in the child env) so we don't
        #      silently invoke `sudo` on a user's machine where it
        #      happens to be passwordless.
        #
        # On a host with neither (e.g. macOS, or a stripped Linux), exec
        # the entrypoint directly -- no containment, but no regression.
        # `MemoryMax=95%` resolves per-host against `MemTotal`, so the
        # cap moves with the runner (16 GB GHA → ~15 GB cap; bigger dev
        # box → bigger cap). Without it the scope has unlimited memory
        # accounting and the OOM-killer fires at the host level, which
        # can reap the GHA runner agent itself -- exit 143, no outcome
        # line. With it, cppad-style "one huge alloc" hits the cgroup
        # limit first, the cgroup OOM-killer reaps only the in-scope
        # process, and the wrapper above it survives to log the failure.
        # 95% leaves enough headroom for the runner-agent + system
        # services while staying close to the effective memory budget
        # the old Docker setup gave each container.
        # The cgroup-scope wrapper is Linux-only: systemd-run, /proc/cgroup,
        # and the systemd service manager don't exist on darwin. Refer to
        # `pkgs.systemd` only inside this Linux-gated block, otherwise the
        # `aarch64-darwin` evaluation refuses (`pkgs.systemd` has no darwin
        # in its `meta.platforms`). On non-Linux we just exec the
        # entrypoint plainly -- same fallback shape as before, no OOM
        # containment.
        ${lib.optionalString pkgs.stdenv.isLinux ''
          # systemd-run and sudo are NOT on the runner's PATH (runtimeInputs
          # is the minimal set the entrypoint needs), so reference both by
          # absolute path. systemd-run is bundled into our closure via
          # nixpkgs's systemd; sudo is a host binary (setuid root, can't
          # be bundled), so probe well-known locations.
          SYSTEMD_RUN=${pkgs.systemd}/bin/systemd-run
          SUDO=""
          for p in /usr/bin/sudo /usr/local/bin/sudo /run/wrappers/bin/sudo; do
            if [ -x "$p" ]; then SUDO="$p"; break; fi
          done

          if [ -x "$SYSTEMD_RUN" ]; then
            if [ -n "''${XDG_RUNTIME_DIR:-}" ] && [ -S "$XDG_RUNTIME_DIR/bus" ]; then
              exec "$SYSTEMD_RUN" --user --scope --quiet --collect \
                -p MemoryMax=95% -p MemorySwapMax=0 -p OOMPolicy=continue \
                -- ${entrypoint} "$@"
            elif [ "''${GRADBENCH_CONTAIN_OOM:-}" = 1 ] \
                 && [ -n "$SUDO" ] && "$SUDO" -n true 2>/dev/null; then
              exec "$SUDO" --preserve-env=PATH \
                "$SYSTEMD_RUN" --scope --quiet --collect \
                --uid="$(id -u)" --gid="$(id -g)" \
                --working-directory="$PWD" \
                -p MemoryMax=95% -p MemorySwapMax=0 -p OOMPolicy=continue \
                -- ${entrypoint} "$@"
            fi
          fi
        ''}

        exec ${entrypoint} "$@"
      '';
    };

  mkEval = { name, ... }@args:
    mkRunner (builtins.removeAttrs args [ ] // { name = "eval-${name}"; });

  mkTool = { name, ... }@args:
    mkRunner (builtins.removeAttrs args [ ] // { name = "tool-${name}"; });

  # A C++-based tool driven by cpp.py, which compiles `tools/<name>/bin/<eval>`
  # on demand (per `define` message) and runs it. The toolchain and AD library
  # must therefore be available at run time; compile-on-demand is intentional,
  # so GradBench can measure each tool's compilation cost.
  #
  #   libs        nixpkgs packages providing headers/libraries (and any
  #               pkg-config files) for the AD library; their include, lib, and
  #               lib/pkgconfig directories are put on the usual search paths.
  #   compiler    the C/C++ compiler package (default gcc).
  #   extraInputs additional tools to put on PATH (e.g. pkg-config, lld).
  #   extraSetup  extra raw shell run before the entrypoint.
  mkCppTool = { name, libs ? [ ], compiler ? pkgs.gcc, extraInputs ? [ ]
    , extraSetup ? "" }:
    let
      # The compile-on-demand binary dynamically links against libstdc++,
      # which doesn't come from `libs` (those are the AD library packages).
      # Use the stdenv compiler's libstdc++ -- nixpkgs's clang-wrapper on
      # Linux also links against this libstdc++, so it's correct for both
      # `pkgs.gcc` and the LLVM clang wrappers we use.
      libraryPathLibs = libs ++ [ pkgs.stdenv.cc.cc.lib ];
      includePath =
        lib.makeSearchPathOutput "dev" "include" ([ jsonInclude ] ++ libs);
      libraryPath = lib.makeLibraryPath libraryPathLibs;
      pkgConfigPath = lib.makeSearchPathOutput "dev" "lib/pkgconfig" libs;
    in mkTool {
      inherit name;
      runtimeInputs = [ pkgs.python3 pkgs.gnumake compiler ] ++ libs
        ++ extraInputs;
      setup = ''
        export CPATH="${includePath}''${CPATH:+:$CPATH}"
        export LIBRARY_PATH="${libraryPath}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${libraryPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export PKG_CONFIG_PATH="${pkgConfigPath}''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
        # The Nix cc-wrapper disables -march=native for build purity. These
        # tools are compiled immediately before running (common.mk defaults
        # NATIVE=yes) and not baked into any image, so re-enable native codegen.
        export NIX_ENFORCE_NO_NATIVE=0
        ${extraSetup}
      '';
      entrypoint = "python3 python/gradbench/gradbench/cpp.py ${name}";
    };

  # A Python eval. All evals are Python and use Nixpkgs' Python packages rather
  # than uv2nix: version fidelity matters little for the harness/validator, and
  # this keeps closures small and cached. numpy, pydantic, and dataclasses-json
  # cover every eval; `pythonPackages` adds any extras (e.g. scipy for gmm).
  mkPyEval = { name, pythonPackages ? (ps: [ ]) }:
    let
      pythonEnv = pkgs.python312.withPackages (ps:
        (with ps; [ numpy pydantic dataclasses-json ]) ++ pythonPackages ps);
    in mkEval {
      inherit name;
      runtimeInputs = [ pythonEnv ];
      setup = ''
        export PYTHONPATH="$root/python/gradbench''${PYTHONPATH:+:$PYTHONPATH}"
      '';
      entrypoint = "python3 python/gradbench/gradbench/evals/${name}/run.py";
    };

  # A Python/ML tool that runs against a uv2nix-built `venv`. The gradbench
  # package is supplied via PYTHONPATH (it is excluded from the venv; see
  # python.nix), matching how the evals work. `extraInputs` adds non-Python
  # programs to PATH (e.g. a compiler), and `extraSetup` runs extra shell.
  mkPyTool = { name, venv, entrypoint, extraInputs ? [ ], extraSetup ? "" }:
    mkTool {
      inherit name entrypoint;
      runtimeInputs = [ venv ] ++ extraInputs;
      setup = ''
        export PYTHONPATH="$root/python/gradbench''${PYTHONPATH:+:$PYTHONPATH}"
        ${extraSetup}
      '';
    };

  # A Julia tool. Each lives in tools/<name> with a Project.toml + Manifest.toml
  # and a relative path dependency on the local julia/GradBench package, which
  # resolves against the checkout. The dependency depot is produced by
  # `Pkg.instantiate()` in a fixed-output derivation; we prune the
  # non-deterministic parts (registries, logs, precompile caches) and keep the
  # content-addressed packages/artifacts. At run time JULIA_DEPOT_PATH points at
  # a writable overlay (for precompile caches) followed by that read-only depot.
  # `depotHashes` is a `{ <system> = "sha256-..."; ... }` attrset, because
  # `Pkg.instantiate()`'s output differs per platform (different precompiled
  # artifacts land in the depot). On a system that has no entry we throw a
  # clear error; the per-tool .nix file is the place to add a new entry once
  # we've computed the hash for that platform.
  mkJuliaTool = { name, depotHashes }:
    let
      julia = pkgs.julia_110;
      depotHash = depotHashes.${pkgs.stdenv.system} or (throw
        "mkJuliaTool ${name}: no depot hash recorded for ${pkgs.stdenv.system}; compute one and add it to depotHashes");
      depot = pkgs.stdenvNoCC.mkDerivation {
        name = "gradbench-julia-${name}-depot";
        inherit src;
        nativeBuildInputs = [ julia pkgs.cacert pkgs.git ];
        dontConfigure = true;
        # Don't let fixupPhase rewrite shebangs in shipped package scripts to
        # /nix/store/...-bash: that would make the depot reference the build
        # shell, which a fixed-output derivation may not do. Julia uses the
        # runtime shell, not these scripts' shebangs.
        dontPatchShebangs = true;
        buildPhase = ''
          export HOME="$TMPDIR"
          export JULIA_DEPOT_PATH="$out"
          export JULIA_PKG_PRECOMPILE_AUTO=0
          julia --project=tools/${name} -e 'import Pkg; Pkg.instantiate()'
        '';
        installPhase = ''
          # Keep only the content-addressed, reproducible parts of the depot.
          rm -rf "$out/registries" "$out/logs" "$out/compiled" \
                 "$out/scratchspaces" || true
        '';
        outputHashMode = "recursive";
        outputHashAlgo = "sha256";
        outputHash = depotHash;
      };
    in mkTool {
      inherit name;
      runtimeInputs = [ julia pkgs.coreutils ];
      setup = ''
        # A persistent writable depot for precompile caches (so we don't
        # recompile every run), with the prebuilt read-only depot underneath.
        # Offline so Julia never reaches the network.
        writable="''${XDG_CACHE_HOME:-$HOME/.cache}/gradbench/julia-${name}"
        mkdir -p "$writable"
        export JULIA_DEPOT_PATH="$writable:${depot}"
        export JULIA_PKG_OFFLINE=true
        # Depot artifacts (e.g. OpenSpecFun_jll's libopenspecfun.so) need
        # libgfortran.so.5 and friends at dlopen time; their RUNPATH is
        # just $ORIGIN so they can't find Julia's bundled copies under
        # lib/julia/ on their own. Put that directory on LD_LIBRARY_PATH.
        export LD_LIBRARY_PATH="${julia}/lib/julia''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      '';
      entrypoint = "julia --project=tools/${name} tools/${name}/run.jl";
    };

  # Build an OCI image from a native runner's closure. The runner already
  # creates a fresh writable workdir from `embeddedSource` at startup, so
  # the image just exposes it directly.
  mkImage = runner:
    pkgs.dockerTools.buildLayeredImage {
      name = "ghcr.io/gradbench/${runner.name}";
      tag = "latest";
      contents = [ pkgs.bashInteractive pkgs.coreutils runner ];
      config = {
        Entrypoint = [ "${runner}/bin/${runner.name}" ];
        Labels."org.opencontainers.image.source" =
          "https://github.com/gradbench/gradbench";
      };
    };

  # Pre-compiled C++ baselines from `tools/manual` that the validating evals
  # (gmm, kmeans, llsq, ba, det, ht, lse, lstm, ode) invoke as a golden
  # reference via `cpp.evaluate(tool="manual", ...)`. The old per-eval
  # Dockerfile did exactly this (`RUN make -C tools/manual -Bj NATIVE=no`);
  # we bake the equivalent into a derivation and overlay its output into
  # `embeddedSource` below. `NATIVE=no` so the baselines are portable across
  # machines, since the derivation is content-addressed and shareable.
  manualBaselines = pkgs.stdenv.mkDerivation {
    name = "gradbench-manual-baselines";
    inherit src;
    nativeBuildInputs = [ pkgs.gnumake ];
    CPATH = "${jsonInclude}/include";
    dontConfigure = true;
    buildPhase = ''
      runHook preBuild
      make -C tools/manual -Bj NATIVE=no
      runHook postBuild
    '';
    installPhase = ''
      runHook preInstall
      mkdir -p "$out"
      cp -r tools/manual/bin "$out/bin"
      runHook postInstall
    '';
  };

  # The repository source embedded into each runner's closure, filtered to
  # the top-level dirs runners actually touch at run time. `python/`, `cpp/`,
  # and `tools/` cover Python tools + C++ compile-on-demand; `js/` is needed
  # by floretta and tensorflow-js; `julia/` is the relative `GradBench`
  # package the Julia tools' `Project.toml`s depend on; `evals/` holds the
  # C++ baseline implementations consumed by validating evals. Keeping this
  # list explicit (rather than copying the whole repo) keeps the per-run
  # tmpdir small. We also overlay `manualBaselines` so the validating evals
  # find `tools/manual/bin/<module>` already compiled, exactly like the old
  # per-eval Docker images did.
  embeddedSource = pkgs.runCommand "gradbench-source" { } ''
    mkdir -p "$out"
    cp -r ${src}/python "$out/python"
    cp -r ${src}/cpp "$out/cpp"
    cp -r ${src}/evals "$out/evals"
    cp -r ${src}/js "$out/js"
    cp -r ${src}/julia "$out/julia"
    cp -r ${src}/tools "$out/tools"
    # Files copied out of the store are read-only; make them writable both so we
    # can drop json.hpp in and so compile-on-demand can write here at run time.
    chmod -R u+w "$out"
    # Provide json.hpp where cpp/Makefile would have downloaded it.
    cp ${jsonInclude}/include/json.hpp "$out/cpp/json.hpp"
    # Overlay the pre-compiled manual baselines.
    mkdir -p "$out/tools/manual/bin"
    cp ${manualBaselines}/bin/* "$out/tools/manual/bin/"
    chmod -R u+w "$out/tools/manual/bin"
  '';
}
