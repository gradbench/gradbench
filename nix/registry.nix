# The registry of all evals and tools. As more are converted from Dockerfiles,
# add them to the `evals` and `tools` attribute sets below; everything else
# (native runners, OCI images, `nix run` apps) is derived automatically.
{ pkgs, pkgsUnstable, src, uv2nix, pyproject-nix, pyproject-build-systems }:

let
  inherit (pkgs) lib;
  gblib = import ./lib.nix { inherit pkgs src; };
  python = import ./python.nix {
    inherit pkgs src uv2nix pyproject-nix pyproject-build-systems;
  };

  eval = name: import (./evals + "/${name}.nix") { inherit pkgs gblib; };
  evals = lib.genAttrs [
    "ba"
    "det"
    "gmm"
    "hello"
    "ht"
    "kmeans"
    "llsq"
    "lse"
    "lstm"
    "ode"
    "particle"
    "saddle"
  ] eval;

  tool = name: import (./tools + "/${name}.nix") { inherit pkgs gblib; };
  # Python/ML tools also receive the uv2nix-built environment.
  pyTool = name:
    import (./tools + "/${name}.nix") { inherit pkgs gblib python; };
  tools = lib.genAttrs [
    # C++-based tools (cpp.py, compile-on-demand).
    "manual"
    "finite"
    "codipack"
    "cppad"
    "adol-c"
    "adept"
    "enzyme"
    "ad-hpp"
    # JavaScript tools.
    "floretta"
    "tensorflow-js"
    # OCaml.
    "ocaml"
    # Julia.
    "zygote"
    "forwarddiff-jl"
    "reversediff-jl"
    "mooncake-jl"
    "enzyme-jl"
    # Haskell.
    "haskell-ad"
  ] tool // lib.genAttrs [
    # Python/ML tools (uv2nix).
    "pytorch"
    "jax"
    "tensorflow"
    "futhark"
    "tapenade"
  ] pyTool;
  # Work in progress, not yet built by default (see each file's header):
  #   tools/scilean.nix   - builds, but the Lake output isn't bit-reproducible.
  #   tools/horde-ad.nix  - needs GHC 9.14 + haskell.nix (uses pkgsUnstable,
  #                         which is why that input is still threaded through).

  prefix = p: set:
    lib.mapAttrs' (name: drv: lib.nameValuePair "${p}${name}" drv) set;
  imagesOf = p: set:
    lib.mapAttrs'
    (name: drv: lib.nameValuePair "image-${p}${name}" (gblib.mkImage drv)) set;
in {
  inherit gblib;
  packages = (prefix "eval-" evals) // (prefix "tool-" tools)
    // (imagesOf "eval-" evals) // (imagesOf "tool-" tools);
}
