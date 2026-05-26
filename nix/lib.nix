# Helpers that capture the patterns previously encoded in the per-eval and
# per-tool Dockerfiles. Each eval/tool becomes a native runner (a wrapper
# script that bakes in dependencies + entrypoint) plus an OCI image built from
# that same runner's closure via `dockerTools`.
{ pkgs, src }:

let inherit (pkgs) lib;
in rec {
  # The nlohmann/json single header, exposed as an include directory so it can
  # be put on CPATH (the C++ sources do `#include "json.hpp"`). This replaces
  # the `wget` in cpp/Makefile.
  jsonInclude = pkgs.runCommand "gradbench-json-include" {
    header = pkgs.fetchurl {
      url =
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp";
      hash = "sha256-m+pMgGbvShwgayvlo2MC+JJvf9xgh69dILQX0M8QPqY=";
    };
  } ''
    mkdir -p "$out/include"
    cp "$header" "$out/include/json.hpp"
  '';

  # A native runner. Equivalent to a Dockerfile's ENTRYPOINT plus the
  # environment it ran in. `setup` is raw shell (e.g. `export` lines) executed
  # before the entrypoint, and `entrypoint` is the command to exec.
  #
  # Native runs operate on the working tree: GRADBENCH_SOURCE_ROOT defaults to
  # the current directory, which is expected to be a GradBench checkout. This
  # keeps compile-on-demand tools writable (they build into tools/<tool>/bin).
  mkRunner = { name, runtimeInputs ? [ ], setup ? "", entrypoint }:
    pkgs.writeShellApplication {
      inherit name runtimeInputs;
      text = ''
        root="''${GRADBENCH_SOURCE_ROOT:-$PWD}"
        if [ ! -d "$root/python/gradbench" ]; then
          echo "gradbench: '$root' does not look like a GradBench checkout;" \
               "set GRADBENCH_SOURCE_ROOT or run from the repository root" >&2
          exit 1
        fi
        ${setup}
        cd "$root"
        exec ${entrypoint} "$@"
      '';
    };

  mkEval = { name, ... }@args:
    mkRunner (builtins.removeAttrs args [ ] // { name = "eval-${name}"; });

  mkTool = { name, ... }@args:
    mkRunner (builtins.removeAttrs args [ ] // { name = "tool-${name}"; });

  # Build an OCI image from a native runner's closure. The repository source is
  # embedded read-only; at startup it is copied to a writable workdir so that
  # compile-on-demand tools can write into tools/<tool>/bin. This is the only
  # place that needs the working tree to be writable.
  mkImage = runner:
    let
      entry = pkgs.writeShellApplication {
        name = "${runner.name}-entrypoint";
        runtimeInputs = [ pkgs.coreutils ];
        text = ''
          work="$(mktemp -d)"
          cp -r --no-preserve=mode,ownership ${embeddedSource}/. "$work/"
          export GRADBENCH_SOURCE_ROOT="$work"
          exec ${runner}/bin/${runner.name} "$@"
        '';
      };
    in pkgs.dockerTools.buildLayeredImage {
      name = "ghcr.io/gradbench/${runner.name}";
      tag = "latest";
      contents = [ pkgs.bashInteractive pkgs.coreutils runner ];
      config = {
        Entrypoint = [ "${entry}/bin/${runner.name}-entrypoint" ];
        Labels."org.opencontainers.image.source" =
          "https://github.com/gradbench/gradbench";
      };
    };

  # The repository source embedded into images, filtered to the files the
  # runners actually need at run time. Keeping this lean keeps images small.
  embeddedSource = pkgs.runCommand "gradbench-source" { } ''
    mkdir -p "$out"
    cp -r ${src}/python "$out/python"
    cp -r ${src}/cpp "$out/cpp"
    cp -r ${src}/evals "$out/evals"
    cp -r ${src}/tools "$out/tools"
    # Files copied out of the store are read-only; make them writable both so we
    # can drop json.hpp in and so compile-on-demand can write here at run time.
    chmod -R u+w "$out"
    # Provide json.hpp where cpp/Makefile would have downloaded it.
    cp ${jsonInclude}/include/json.hpp "$out/cpp/json.hpp"
  '';
}
