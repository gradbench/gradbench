# shell.nix for use with Nix, in order to run GradBench locally.
#
# The Nixpkgs snapshot is pinned with niv. Use
#
#  $ niv update nixpkgs -b nixos-unstable
#
# to update it. You can also pass another branch than nixos-unstable,
# but nixos-unstable strikes a nice balance of being pretty new while
# also being cached upstream. Recompiling the Nix world from scratch
# is reliable, but time-consuming.
#
# The nix/ directory contains some derivations that have yet to be
# upstreamed to Nixpkgs, as well as the niv configuration.
let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs { };

  cppad = pkgs.callPackage ./nix/cppad.nix { };
  adept = pkgs.callPackage ./nix/adept.nix { };
  codipack = pkgs.callPackage ./nix/codipack.nix { };
  floretta = pkgs.callPackage ./nix/floretta.nix { };
  GRADBENCH_PATH = builtins.getEnv "PWD";

  isX86 = builtins.currentSystem == "x86_64-linux";
in pkgs.stdenv.mkDerivation rec {
  name = "gradbench";
  buildInputs = [
    # Required
    pkgs.bun
    pkgs.cargo
    pkgs.niv
    pkgs.python311
    pkgs.uv

    pkgs.llvmPackages_19.clang-tools # Must come before clang for clangd to work.

    # Convenient
    pkgs.adolc
    pkgs.blas
    pkgs.eigen
    pkgs.enzyme
    pkgs.futhark
    pkgs.gh
    pkgs.lapack
    pkgs.llvmPackages_19.clang
    pkgs.llvmPackages_19.lld
    pkgs.nixfmt-classic
    pkgs.nodejs_23
    pkgs.openblas
    pkgs.pkg-config
    pkgs.wasm-tools
    pkgs.wget
    pkgs.zlib

    # Custom
    adept
    cppad
    codipack
    floretta

    # Haskell
    pkgs.cabal-install
    pkgs.ghc

    # Rust
    pkgs.clippy
    pkgs.rustc
    pkgs.rustfmt
    pkgs.rust-analyzer

    # OCaml
    pkgs.opam
    pkgs.ocamlPackages.dune_3
    pkgs.ocamlPackages.ocaml
  ] ++
    # Nixpkgs marks Julia as broken on Apple Silicon
    (if isX86 then [ pkgs.julia_110 ] else [ ]);

  # The following are environment variables used by various tools.
  RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
  ENZYME_LIB = "${pkgs.enzyme}/lib";
  LD_LIBRARY_PATH =
    "${pkgs.lib.makeLibraryPath buildInputs}:${pkgs.stdenv.cc.cc.lib}/lib";
}
