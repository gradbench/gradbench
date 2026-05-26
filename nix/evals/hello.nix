# The `hello` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "hello"; }
