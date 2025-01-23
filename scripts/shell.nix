# shell.nix for use with Nix, in order to run Gradbench locally.
#
# May get out of sync with in particular the poetry config (I couldn't
# figure out how to import it automatically), but hopefully it is not
# too difficult to fix.
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
  ];
  my-python = pkgs.python3.withPackages my-python-packages;
in
pkgs.stdenv.mkDerivation {
  name = "gradbench";
  buildInputs =
    [my-python
     pkgs.cargo
     pkgs.rustc
     pkgs.rustfmt
     pkgs.futhark
     pkgs.enzyme
     pkgs.adolc
     pkgs.llvmPackages_19.lld
     pkgs.llvmPackages_19.clang
    ];
}
