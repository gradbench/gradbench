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
  };

  outputs = { self, nixpkgs }:
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
      packages = forAllSystems ({ pkgs, ... }:
        (import ./nix/registry.nix {
          inherit pkgs;
          src = self;
        }).packages);

      # `nix run .#eval-hello`, `nix run .#tool-manual`, etc.
      apps = forAllSystems ({ pkgs, system }:
        let
          registry = import ./nix/registry.nix {
            inherit pkgs;
            src = self;
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
