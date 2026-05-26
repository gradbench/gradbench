# JAX (CPU). Mirrors the Dockerfile's groups (gradbench, jax); gradbench's own
# deps (numpy/pydantic/dataclasses-json) come via commonDeps + PYTHONPATH.
{ pkgs, gblib, python }:

gblib.mkPyTool {
  name = "jax";
  venv = python.venvFor "jax" (python.commonDeps ++ [ "jax" ]);
  entrypoint = "python python/gradbench/gradbench/tools/jax/run.py";
}
