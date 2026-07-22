# The `lstm` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "lstm"; }
