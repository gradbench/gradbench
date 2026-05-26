# The OCaml tool. Unlike the C++ tools it is built ahead of time (as in its
# Dockerfile) into a single `gradbench` binary, here via buildDunePackage with
# owl and yojson from Nixpkgs (owl 1.2 matches the Dockerfile's pin).
{ pkgs, gblib }:

let
  op = pkgs.ocamlPackages;
  gradbench-ocaml = op.buildDunePackage {
    pname = "gradbench";
    version = "0";
    src = ../../tools/ocaml;
    buildInputs = [ op.owl op.yojson ];
  };
in gblib.mkTool {
  name = "ocaml";
  runtimeInputs = [ gradbench-ocaml ];
  entrypoint = "gradbench";
}
