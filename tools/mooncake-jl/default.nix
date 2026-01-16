{ pkgs, src, pythonSet }:
let
  common = import ../../nix/common.nix { inherit pkgs src pythonSet; };
in
common.mkJuliaTool { name = "mooncake-jl"; }
