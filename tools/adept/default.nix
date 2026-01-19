{ pkgs, src, pythonSet }:
let
  common = import ../../nix/common.nix { inherit pkgs src pythonSet; };
in
common.mkCppTool {
  name = "adept";
  extraInputs = [ pkgs.adept ];
}
