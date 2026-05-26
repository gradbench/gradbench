# The `det` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "det"; }
