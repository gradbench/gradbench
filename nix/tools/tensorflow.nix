# TensorFlow (CPU). Mirrors the Dockerfile's groups: dataclasses-json, scipy,
# tensorflow (gradbench via PYTHONPATH).
{ pkgs, gblib, python }:

gblib.mkPyTool {
  name = "tensorflow";
  venv =
    python.venvFor "tensorflow" (python.commonDeps ++ [ "scipy" "tensorflow" ]);
  entrypoint = "python python/gradbench/gradbench/tools/tensorflow/run.py";
}
