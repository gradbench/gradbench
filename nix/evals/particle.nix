# The `particle` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "particle"; }
