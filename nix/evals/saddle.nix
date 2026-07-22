# The `saddle` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "saddle"; }
