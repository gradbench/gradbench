{ pkgs, src, pythonSet }:
let
  common = import ../../nix/common.nix { inherit pkgs src pythonSet; };
in
common.mkCppTool {
  name = "adol-c";
  extraInputs = [ pkgs.adolc ];
}
