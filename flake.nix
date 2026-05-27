{
  description = "GradBench: a benchmark suite for differentiable programming";

  # The Nixpkgs revision is pinned in flake.lock. It matches the revision that
  # was previously pinned with niv in nix/sources.json, so that we reuse the
  # same binary cache. To update:
  #
  #   $ nix flake update nixpkgs
  #
  # nixos-unstable strikes a balance between being recent and being cached
  # upstream.
  inputs = {
    nixpkgs.url =
      "github:NixOS/nixpkgs/d89fc19e405cb2d55ce7cc114356846a0ee5e956";

    # A newer Nixpkgs used ONLY for tools that need a toolchain the pinned
    # Nixpkgs lacks (currently horde-ad, which needs GHC 9.14). Keeping it
    # separate avoids rebuilding/revalidating every other tool against a new
    # pin. Everything else uses `nixpkgs` above.
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";

    # haskell.nix builds Haskell projects directly from their cabal.project,
    # honoring the pinned Hackage index-state and providing GHCs (incl. 9.14).
    # Used ONLY for horde-ad, whose plugin dependencies aren't 9.14-ready in
    # Nixpkgs. Best used with IOG's binary cache (https://cache.iog.io).
    haskellNix.url = "github:input-output-hk/haskell.nix";

    # lean4-nix builds Lake projects with Nix: it installs the pinned Lean
    # toolchain and turns lake-manifest.json + the Lake build graph into normal
    # Nix derivations. Used ONLY for scilean. (Its deps, incl. mathlib, are
    # built from source, since mathlib isn't in Nixpkgs.)
    lean4-nix.url = "github:lenianiva/lean4-nix";

    # uv2nix and friends build hermetic Python environments from uv.lock, used
    # for the ML/Python tools (pytorch, jax, tensorflow, ...). Evals use plain
    # Nixpkgs Python instead.
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ self, nixpkgs, ... }:
    let
      # The proof-of-concept currently targets x86_64-linux only. Adding
      # aarch64-linux and aarch64-darwin later should just mean extending this
      # list (and resolving any per-tool platform-specific logic).
      systems = [ "x86_64-linux" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f {
            inherit system;
            pkgs = import nixpkgs { inherit system; };
            pkgsUnstable = import inputs.nixpkgs-unstable { inherit system; };
            # haskell.nix-enabled pkgs (its overlay on its own nixpkgs), for
            # horde-ad only.
            pkgsHaskellNix = import inputs.haskellNix.inputs.nixpkgs {
              inherit system;
              inherit (inputs.haskellNix) config;
              overlays = [ inputs.haskellNix.overlay ];
            };
          });
    in {
      # Each eval and tool is exposed as a runnable package:
      #
      #   packages.<system>.eval-<name>    native wrapper (run on the host)
      #   packages.<system>.tool-<name>    native wrapper (run on the host)
      #   packages.<system>.image-eval-<name>   OCI image (dockerTools)
      #   packages.<system>.image-tool-<name>   OCI image (dockerTools)
      #
      # The native wrapper and the OCI image are two outputs of the same
      # underlying derivation: the wrapper bakes in the dependencies and
      # entrypoint that used to live in the tool's Dockerfile.
      packages = forAllSystems ({ pkgs, pkgsUnstable, pkgsHaskellNix, system }:
        (import ./nix/registry.nix {
          inherit pkgs pkgsUnstable pkgsHaskellNix system;
          src = self;
          inherit (inputs)
            uv2nix pyproject-nix pyproject-build-systems lean4-nix;
        }).packages);

      # `nix run .#eval-hello`, `nix run .#tool-manual`, etc.
      apps = forAllSystems ({ pkgs, pkgsUnstable, pkgsHaskellNix, system }:
        let
          registry = import ./nix/registry.nix {
            inherit pkgs pkgsUnstable pkgsHaskellNix system;
            src = self;
            inherit (inputs)
              uv2nix pyproject-nix pyproject-build-systems lean4-nix;
          };
        in builtins.mapAttrs (name: pkg: {
          type = "app";
          program = "${pkg}/bin/${name}";
        }) (nixpkgs.lib.filterAttrs
          (name: _: !(nixpkgs.lib.hasPrefix "image-" name)) registry.packages));

      # Replaces shell.nix. Enter with `nix develop` (or via direnv).
      devShells = forAllSystems ({ pkgs, ... }: {
        default = import ./nix/devshell.nix { inherit pkgs; };
      });

      formatter = forAllSystems ({ pkgs, ... }: pkgs.nixfmt-classic);
    };
}
