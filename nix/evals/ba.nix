# The `ba` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "ba"; }
