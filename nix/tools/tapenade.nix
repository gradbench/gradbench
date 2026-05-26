# Tapenade (INRIA), a source-to-source AD tool. For most evals run.py uses
# cpp.py to compile pre-generated derivative C/C++ (compile-on-demand); for
# `hello` it invokes the Tapenade binary live. Tapenade is referenced by the
# relative path `tapenade_3.16/...`, so we materialize the distribution there
# (like ad-hpp/futhark). It is a Java program, so we provide a JDK.
{ pkgs, gblib, python }:

let
  tapenadeDist = pkgs.stdenvNoCC.mkDerivation {
    pname = "tapenade";
    version = "3.16";
    src = pkgs.fetchurl {
      url =
        "https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar";
      hash = "sha256-wzqVI8J66cP8cxPX+TSF1/gfAWu3Ga0hR8Ej2uCqUlw=";
    };
    dontConfigure = true;
    dontBuild = true;
    installPhase = ''
      runHook preInstall
      mkdir -p "$out"
      cp -r . "$out/"
      runHook postInstall
    '';
  };
in gblib.mkPyTool {
  name = "tapenade";
  venv = python.venvFor "tapenade" python.commonDeps;
  extraInputs = [ pkgs.jdk17 pkgs.gcc pkgs.gnumake pkgs.coreutils ];
  extraSetup = ''
    export CPATH="${gblib.jsonInclude}/include''${CPATH:+:$CPATH}"
    ln -sfn ${tapenadeDist} "$root/tapenade_3.16"
  '';
  entrypoint = "python python/gradbench/gradbench/tools/tapenade/run.py";
}
