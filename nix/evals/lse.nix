# The `lse` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "lse"; }
