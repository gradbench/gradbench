# PyTorch. Runs against a uv2nix-built venv (CPU torch; see nix/python.nix and
# the pytorch-cpu index in pyproject.toml). Mirrors the Dockerfile's groups:
# dataclasses-json, numpy, scipy, torch (gradbench comes via PYTHONPATH).
{ pkgs, gblib, python }:

gblib.mkPyTool {
  name = "pytorch";
  venv = python.venvFor "pytorch" (python.commonDeps ++ [ "scipy" "torch" ]);
  entrypoint = "python python/gradbench/gradbench/tools/pytorch/run.py";
}
