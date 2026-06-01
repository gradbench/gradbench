# Hermetic Python environments built from uv.lock with uv2nix, for the Python/ML
# tools (pytorch, jax, tensorflow, ...). The lockfile routes torch through the
# PyTorch CPU index (see pyproject.toml), so this pulls CPU wheels rather than
# the multi-gigabyte CUDA ones.
#
# Evals do NOT use this; they use plain Nixpkgs Python packages (see mkPyEval).
{ pkgs, src, uv2nix, pyproject-nix, pyproject-build-systems }:

let
  inherit (pkgs) lib;
  python = pkgs.python312;

  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = src; };

  # Prefer prebuilt wheels (the ML frameworks ship manylinux wheels).
  overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

  # Fix-ups for wheels that autoPatchelf can't fully resolve on its own.
  overrides = final: prev: {
    # Its shared library links libtensorflow_framework.so.2, which lives in the
    # separate `tensorflow` package and is resolved at run time, not at
    # patch time. Tell autoPatchelf not to fail on it.
    tensorflow-io-gcs-filesystem =
      prev.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
        autoPatchelfIgnoreMissingDeps =
          (old.autoPatchelfIgnoreMissingDeps or [ ])
          ++ [ "libtensorflow_framework.so.2" ];
      });
  };

  pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
    inherit python;
  }).overrideScope (lib.composeManyExtensions [
    pyproject-build-systems.overlays.default
    overlay
    overrides
  ]);

  # Build a venv from an explicit list of third-party package names. We
  # deliberately do NOT include any local/meta package (the `gradbench` package
  # would drag in the hatchling build backend -> trove-classifiers -> calver,
  # which needs a newer setuptools than the build bootstrap). Like the evals,
  # gradbench is supplied via PYTHONPATH (see mkPyTool). Per-tool venvs mirror
  # the per-tool dependency groups the Dockerfiles installed.
  venvFor = name: packages:
    pythonSet.mkVirtualEnv "gradbench-${name}-env"
    (lib.genAttrs packages (_: [ ]));
in {
  inherit venvFor;
  # Dependencies shared by the gradbench harness code these tools import.
  commonDeps = [ "numpy" "pydantic" "dataclasses-json" ];
}
