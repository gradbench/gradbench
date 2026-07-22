# The `gmm` eval, which additionally uses SciPy.
{ pkgs, gblib }:

gblib.mkPyEval {
  name = "gmm";
  pythonPackages = ps: [ ps.scipy ];
}
