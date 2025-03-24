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
  GRADBENCH_PATH = builtins.getEnv "PWD";
in pkgs.stdenv.mkDerivation rec {
  name = "gradbench";
  buildInputs = [
    pkgs.bun
    pkgs.gh
    pkgs.niv
    pkgs.nixfmt-classic
    pkgs.python311
    pkgs.uv

    pkgs.futhark
    pkgs.enzyme
    pkgs.pkg-config
    pkgs.llvmPackages_19.lld
    pkgs.llvmPackages_19.clang
    pkgs.blas
    pkgs.lapack
    pkgs.openblas
    pkgs.zlib
    pkgs.adolc
    pkgs.eigen
    pkgs.wget
    adept
    cppad
    codipack

    # Haskell
    pkgs.cabal-install
    pkgs.ghc

    # Rust
    pkgs.cargo
    pkgs.clippy
    pkgs.rustc
    pkgs.rustfmt

    # OCaml
    pkgs.opam
    pkgs.ocamlPackages.dune_3
    pkgs.ocamlPackages.ocaml
  ];

  # The following are environment variables used by various tools.
  RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
  ENZYME_LIB = "${pkgs.enzyme}/lib";
  LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}:${pkgs.stdenv.cc.cc.lib}/lib";
}
