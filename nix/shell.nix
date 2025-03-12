# shell.nix for use with Nix, in order to run Gradbench locally.
#
# May get out of sync with in particular pyproject.toml (I couldn't
# figure out how to import it automatically), but hopefully it is not
# too difficult to maintain.
{ pkgs ? import <nixpkgs> {} }:
let
  my-python-packages = ps: with ps; [
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
  my-python = pkgs.python3.withPackages my-python-packages;
  cppad = pkgs.callPackage ./cppad.nix {};
  adept = pkgs.callPackage ./adept.nix {};
  codipack = pkgs.callPackage ./codipack.nix {};
  GRADBENCH_PATH = builtins.getEnv "PWD";
in
pkgs.stdenv.mkDerivation {
  name = "gradbench";
  buildInputs =
    [my-python
     pkgs.niv
     pkgs.gh
     pkgs.ruff

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
     pkgs.rustc
     pkgs.rustfmt

     # OCaml
     pkgs.opam
     pkgs.ocamlPackages.dune_3
     pkgs.ocamlPackages.ocaml
     pkgs.ocamlPackages.owl
     pkgs.ocamlPackages.yojson
     pkgs.ocamlPackages.findlib
    ];

  # The following are environment variables used by various tools.
  PYTHONPATH = "${GRADBENCH_PATH}/python/gradbench";
  ENZYME_LIB = "${pkgs.enzyme}/lib";
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
