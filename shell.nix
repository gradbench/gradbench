# shell.nix for use with Nix, in order to run GradBench locally.
#
# May get out of sync with the uv setup, but hopefully it is not too
# difficult to maintain manually.
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

  gradbench-python-packages = ps:
    with ps; [
      (buildPythonPackage rec {
        pname = "futhark-data";
        version = "1.0.2";
        src = fetchPypi {
          inherit pname version;
          sha256 = "sha256-FJOhVr65U3kP408BbA42jbGGD6x+tVh+TNhsYv8bUT0=";
        };
        doCheck = false;
      })
      (buildPythonPackage rec {
        pname = "futhark-server";
        version = "1.0.0";
        src = fetchPypi {
          inherit pname version;
          sha256 = "sha256-I2+8BeEOPOV0fXtrXdz/eqCj8DFAJTbKmUKy43oiyjE=";
        };
        doCheck = false;
      })
      numpy
      termcolor
      black
      isort
      pytorch
      jax
      jaxlib
      tensorflow
      autograd
      dataclasses-json
      pydantic
      matplotlib
    ];
  gradbench-python = pkgs.python3.withPackages gradbench-python-packages;
  cppad = pkgs.callPackage ./nix/cppad.nix { };
  adept = pkgs.callPackage ./nix/adept.nix { };
  codipack = pkgs.callPackage ./nix/codipack.nix { };
  GRADBENCH_PATH = builtins.getEnv "PWD";
in pkgs.stdenv.mkDerivation {
  name = "gradbench";
  buildInputs = [
    gradbench-python
    pkgs.bun
    pkgs.niv
    pkgs.gh
    pkgs.ruff
    pkgs.nixfmt

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
    adept
    cppad
    codipack

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
  PYTHONPATH = "${GRADBENCH_PATH}/python/gradbench";
  RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
  ENZYME_LIB = "${pkgs.enzyme}/lib";
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
