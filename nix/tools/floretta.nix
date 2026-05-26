# Floretta: a reverse-mode AD tool for WebAssembly. The runner is a Node script
# (run via Node's TypeScript support). The Dockerfile downloaded architecture-
# specific `floretta` and `wasm-tools` binaries; we take both from Nix instead
# (floretta via nix/floretta.nix, wasm-tools from Nixpkgs).
#
# The JS has no external npm dependencies: it only imports the local workspace
# package `@gradbench/common`, so rather than a hermetic node_modules we just
# create the workspace symlinks that `bun install` would have made.
{ pkgs, gblib }:

let floretta = pkgs.callPackage ../floretta.nix { };
in gblib.mkTool {
  name = "floretta";
  runtimeInputs = [ pkgs.nodejs_24 pkgs.wasm-tools floretta pkgs.coreutils ];
  setup = ''
    mkdir -p "$root/node_modules/@gradbench"
    ln -sfn "$root/js/common" "$root/node_modules/@gradbench/common"
    ln -sfn "$root/js/floretta" "$root/node_modules/@gradbench/floretta"
  '';
  entrypoint = "node --disable-warning=ExperimentalWarning js/floretta/run.ts";
}
