{ pkgs, src, pythonSet }:
let
  common = import ../../nix/common.nix { inherit pkgs src pythonSet; };
in
common.mkScileanTool { name = "scilean"; }
