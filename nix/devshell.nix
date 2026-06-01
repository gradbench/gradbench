# Development shell for working on GradBench, used via `nix develop` (and direnv
# `use flake`). This is the flake-based successor to the old niv-pinned
# shell.nix; it provides the dependencies needed for most evals and tools.
{ pkgs }:

let
  floretta = pkgs.callPackage ./floretta.nix { };
  isX86 = pkgs.stdenv.hostPlatform.system == "x86_64-linux";
  packages = [
    # Required
    pkgs.bun
    pkgs.cargo
    pkgs.python312
    pkgs.uv

    pkgs.llvmPackages_19.clang-tools # Must come before clang for clangd to work.

    # Convenient
    pkgs.adept
    pkgs.adolc
    pkgs.blas
    pkgs.codipack
    pkgs.cppad
    pkgs.eigen
    pkgs.enzyme
    pkgs.futhark
    pkgs.gh
    pkgs.lapack
    pkgs.llvmPackages_19.clang
    pkgs.llvmPackages_19.lld
    pkgs.llvmPackages_19.openmp
    pkgs.nix-output-monitor # CLI auto-detects and uses `nom` for prettier builds.
    pkgs.nixfmt-classic
    pkgs.nodejs_24
    pkgs.openblas
    pkgs.pkg-config
    pkgs.wasm-tools
    pkgs.wget
    pkgs.zlib

    # Custom
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
    # Nixpkgs marks Julia as broken on Apple Silicon.
    (pkgs.lib.optionals isX86 [ pkgs.julia_110 ]);
in pkgs.mkShell {
  name = "gradbench";
  inherit packages;

  # Environment variables used by various tools.
  RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
  ENZYME_LIB = "${pkgs.enzyme}/lib";
  LD_LIBRARY_PATH =
    "${pkgs.lib.makeLibraryPath packages}:${pkgs.stdenv.cc.cc.lib}/lib";

  # The Nix C/C++ compilers disable -march=native for purity, but we don't use
  # them to compile Nix derivations in this shell.
  NIX_ENFORCE_NO_NATIVE = 0;

  shellHook = ''
    export GRADBENCH_SOURCE_ROOT="$PWD"
  '';
}
