# TensorFlow.js. A Node tool that needs a real node_modules (@tensorflow/tfjs,
# pure JS). We build it with `bun install` in a fixed-output derivation (as the
# Dockerfile does) and place it at js/tensorflow/node_modules, where Node's
# module resolution finds it from run.ts. The tfjs CPU backend is pure
# JavaScript, so node_modules has no native builds.
{ pkgs, gblib }:

let
  nodeModules = pkgs.stdenvNoCC.mkDerivation {
    name = "gradbench-tfjs-node-modules";
    src = ../..;
    nativeBuildInputs = [ pkgs.bun pkgs.cacert ];
    dontConfigure = true;
    # Don't let fixupPhase rewrite package bin shebangs to /nix/store/...-bash;
    # a fixed-output derivation may not reference store paths.
    dontFixup = true;
    buildPhase = ''
      export HOME="$TMPDIR"
      bun install --omit dev
    '';
    installPhase = ''
      cp -R node_modules "$out"
    '';
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = "sha256-+yTFLYzY/Czc4sytTdezcPCYW22XQEF5yQqp/kNiiKU=";
  };
in gblib.mkTool {
  name = "tensorflow-js";
  runtimeInputs = [ pkgs.nodejs_24 pkgs.coreutils ];
  setup = ''
    # node_modules must be a real directory (not a symlink to the store): Node
    # realpaths symlinks, and inter-package resolution (tfjs -> tfjs-core) only
    # works when the packages sit under a directory literally named
    # node_modules. We materialize the third-party packages from the store once
    # (cached by the store path), and point @gradbench/common at the live source
    # so it resolves OUTSIDE node_modules (Node won't type-strip .ts under
    # node_modules).
    nm="$root/js/tensorflow/node_modules"
    if [ "$(cat "$nm/.gradbench-store" 2>/dev/null)" != "${nodeModules}" ]; then
      rm -rf "$nm"
      mkdir -p "$nm"
      cp -R --no-preserve=mode,ownership ${nodeModules}/. "$nm/"
      rm -rf "$nm/@gradbench"
      mkdir -p "$nm/@gradbench"
      ln -sfn "$root/js/common" "$nm/@gradbench/common"
      echo "${nodeModules}" > "$nm/.gradbench-store"
    fi
  '';
  entrypoint =
    "node --disable-warning=ExperimentalWarning js/tensorflow/run.ts";
}
