# horde-ad. WORK IN PROGRESS, not yet wired into the registry.
#
# Needs GHC 9.14, which only the newer Nixpkgs (pkgsUnstable) has. horde-ad and
# ox-arrays are present there but marked broken, and the GHC typelits plugins
# they depend on carry stale `ghc < 9.13` bounds. Unbreaking + jailbreaking gets
# past those, but the plugin sources themselves do not build on GHC 9.14 in this
# snapshot (e.g. ghc-tcplugins-extra 0.5: "can't find source for Internal") --
# which is why Nixpkgs marks the whole stack broken on 9.14.
#
# The blocker: callCabal2nix resolves against the Nixpkgs package set, so it
# can't honor the cabal.project's pinned Hackage `index-state` (2026-04-15),
# where 9.14-compatible plugin versions presumably live. Producing horde-ad
# really wants `haskell.nix` (which resolves a consistent plan from Hackage at
# that index-state). The override scaffolding below is kept as a starting point.
{ pkgs, pkgsUnstable, gblib }:

let
  hl = pkgsUnstable.haskell.lib;
  hp = pkgsUnstable.haskell.packages.ghc914.override {
    overrides = self: super: {
      ox-arrays = hl.markUnbroken super.ox-arrays;
      horde-ad = hl.doJailbreak (hl.markUnbroken super.horde-ad);
      # These carry stale `ghc < 9.13` upper bounds; relax them (the Dockerfile
      # builds with --allow-newer for the same reason).
      ghc-tcplugins-extra = hl.doJailbreak super.ghc-tcplugins-extra;
      ghc-typelits-knownnat = hl.doJailbreak super.ghc-typelits-knownnat;
      ghc-typelits-natnormalise =
        hl.doJailbreak super.ghc-typelits-natnormalise;
    };
  };
  gradbench-hs =
    hl.doJailbreak (hp.callCabal2nix "gradbench" ../../tools/horde-ad { });
in gblib.mkTool {
  name = "horde-ad";
  runtimeInputs = [ gradbench-hs ];
  entrypoint = "gradbench";
}
