# Tapenade (INRIA), a source-to-source AD tool. For most evals run.py uses
# cpp.py to compile pre-generated derivative C/C++ (compile-on-demand); for
# `hello` it invokes the Tapenade binary live. Tapenade is referenced by the
# relative path `tapenade_3.16/...`, so we materialize the distribution there
# (like ad-hpp/futhark). It is a Java program, so we provide a JDK.
#
# We pull the tarball from the upstream GitLab generic package registry
# rather than `tapenade.gitlabpages.inria.fr/distrib/tapenade_3.16.tar`,
# because the latter is a moving URL: INRIA silently re-uploads it (we saw
# the hash flip during a CI run) and a fixed-output `fetchurl` can't survive
# that. The generic-packages URL is keyed by version (3.16.2 here) and is
# immutable per version, so the hash is stable.
{ pkgs, gblib, python }:

let
  tapenadeDist = pkgs.stdenvNoCC.mkDerivation {
    pname = "tapenade";
    version = "3.16.2";
    src = pkgs.fetchurl {
      url =
        "https://gitlab.inria.fr/api/v4/projects/tapenade%2Ftapenade/packages/generic/tapenade/3.16.2/tapenade_3.16.2.tar";
      hash = "sha256-V5osgh9TQWVxzxDsve8r2Tdf4wkPAyjJhjMtdTm4nXc=";
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
