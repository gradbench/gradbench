# haskell-ad. Built ahead of time with callCabal2nix against GHC 9.12 (matching
# the Dockerfile's haskell:9.12); its deps (ad, aeson, vector, ...) are all in
# Nixpkgs. cabal.project pins `ad +ffi`, so we enable that flag on the ad dep.
{ pkgs, gblib }:

let
  hl = pkgs.haskell.lib;
  hp = pkgs.haskell.packages.ghc912.override {
    overrides = self: super: { ad = hl.enableCabalFlag super.ad "ffi"; };
  };
  gradbench-hs = hp.callCabal2nix "gradbench" ../../tools/haskell-ad { };
in gblib.mkTool {
  name = "haskell-ad";
  runtimeInputs = [ gradbench-hs ];
  entrypoint = "gradbench";
}
