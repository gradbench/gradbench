# The `llsq` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "llsq"; }
