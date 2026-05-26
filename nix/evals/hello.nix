# The `hello` eval. Like all evals it is Python; the harness only needs numpy,
# pydantic, and dataclasses-json (see README "Running evals outside of
# Docker"). Evals use nixpkgs' Python packages rather than uv2nix: version
# fidelity matters little for the harness/validator, and this keeps the closure
# small and cached.
{ pkgs, gblib }:

let
  pythonEnv = pkgs.python311.withPackages
    (ps: with ps; [ numpy pydantic dataclasses-json ]);
in gblib.mkEval {
  name = "hello";
  runtimeInputs = [ pythonEnv ];
  setup = ''
    export PYTHONPATH="$root/python/gradbench''${PYTHONPATH:+:$PYTHONPATH}"
  '';
  entrypoint = "python3 python/gradbench/gradbench/evals/hello/run.py";
}
