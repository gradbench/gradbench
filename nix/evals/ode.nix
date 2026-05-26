# The `ode` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "ode"; }
