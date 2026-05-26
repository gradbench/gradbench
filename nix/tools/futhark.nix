# Futhark. Mirrors the Dockerfile's groups (gradbench, dataclasses-json,
# futhark-server) plus the Futhark compiler from Nixpkgs and a C compiler for
# the (compile-on-demand) `c` backend.
#
# `futhark pkg sync` downloads the pinned packages in tools/futhark/futhark.pkg.
# We run it in a fixed-output derivation (it needs network), then symlink the
# result into tools/futhark/lib where the .fut programs import it.
{ pkgs, gblib, python }:

let
  # Nixpkgs has Futhark 0.25.29, but GradBench pins 0.26.1 (0.25.x has AD
  # compiler bugs that break some evals). Use the official binary release, as
  # the Dockerfile does.
  futhark = pkgs.stdenv.mkDerivation rec {
    pname = "futhark";
    version = "0.26.1";
    src = pkgs.fetchurl {
      url =
        "https://futhark-lang.org/releases/futhark-${version}-linux-x86_64.tar.xz";
      hash = "sha256-3VZs7QwJkEqwbrU0RmGqM79cPRCMz19LTQQXlYQGXJ0=";
    };
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib pkgs.gmp pkgs.zlib ];
    dontConfigure = true;
    dontBuild = true;
    installPhase = ''
      runHook preInstall
      mkdir -p "$out"
      cp -r bin share "$out/"
      runHook postInstall
    '';
  };

  futharkLib = pkgs.stdenvNoCC.mkDerivation {
    name = "gradbench-futhark-pkgs";
    src = ../../tools/futhark;
    nativeBuildInputs = [ futhark pkgs.cacert pkgs.git ];
    dontConfigure = true;
    buildPhase = ''
      export HOME="$TMPDIR"
      futhark pkg sync
    '';
    installPhase = ''
      mkdir -p "$out"
      cp -r lib/. "$out/"
    '';
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-TecIgC34DMnAJgVgSniplZUWt1iTBgcljZXfzlWELC8=";
  };
in gblib.mkPyTool {
  name = "futhark";
  venv = python.venvFor "futhark" (python.commonDeps ++ [ "futhark-server" ]);
  extraInputs = [ futhark pkgs.gcc pkgs.coreutils ];
  extraSetup = ''
    ln -sfn ${futharkLib} "$root/tools/futhark/lib"
  '';
  entrypoint = "python tools/futhark/run.py";
}
