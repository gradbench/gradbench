# horde-ad. Needs GHC 9.14 and the horde-ad/ox-arrays libraries plus GHC
# typelits plugins that aren't 9.14-ready in Nixpkgs. Built with haskell.nix,
# which reads tools/horde-ad/cabal.project directly -- honoring its pinned
# Hackage index-state, so it picks 9.14-compatible plugin versions -- and
# provides GHC 9.14. The build plan resolves cleanly (ghc9141 accepted,
# index-state honored).
#
# IMPORTANT: this requires IOG's binary cache, or GHC 9.14 is built from source
# (hours). Add to your nix.conf (as a trusted user):
#   extra-substituters = https://cache.iog.io
#   extra-trusted-public-keys = hydra.iohk.io:f/Ea+s+dFdN+3Y/G+FDgSq+a5NEWhJGzdjvKNGv0/EQ=
# Then `nix build .#tool-horde-ad` works.
{ pkgs, pkgsHaskellNix, gblib }:

let
  project = pkgsHaskellNix.haskell-nix.cabalProject' {
    src = ../../tools/horde-ad;
    compiler-nix-name = "ghc9141";
    # The Dockerfile builds with `cabal --allow-newer`; some transitive deps
    # (e.g. criterion, microstache) carry stale `base < 4.22` bounds that
    # otherwise make the GHC 9.14 plan unsolvable. Relax all bounds likewise.
    cabalProjectLocal = "allow-newer: all";
  };
  gradbench-hs = project.hsPkgs.gradbench.components.exes.gradbench;
in gblib.mkTool {
  name = "horde-ad";
  runtimeInputs = [ gradbench-hs ];
  entrypoint = "gradbench";
}
