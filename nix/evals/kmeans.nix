# The `kmeans` eval.
{ pkgs, gblib }:

gblib.mkPyEval { name = "kmeans"; }
