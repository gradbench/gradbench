# The registry of all evals and tools. As more are converted from Dockerfiles,
# add them to the `evals` and `tools` attribute sets below; everything else
# (native runners, OCI images, `nix run` apps) is derived automatically.
{ pkgs, pkgsHaskellNix, system, src, uv2nix, pyproject-nix
, pyproject-build-systems, lean4-nix }:

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
  ] pyTool // {
    # horde-ad: converted via haskell.nix. NOTE: building it requires IOG's
    # binary cache (https://cache.iog.io) or GHC 9.14 is built from source; see
    # nix/tools/horde-ad.nix. The build plan resolves; not built in CI yet.
    horde-ad =
      import ./tools/horde-ad.nix { inherit pkgs pkgsHaskellNix gblib; };
    # scilean: built with lean4-nix (its deps, incl. mathlib, from source).
    scilean = import ./tools/scilean.nix { inherit lean4-nix system gblib; };
  };

  # Which systems each tool actually supports. Tools not listed default to
  # all of `flake.nix`'s `systems`; tools listed are restricted to the given
  # set and dropped from the registry on other systems. Keep this list short
  # and document the reason next to each entry.
  toolSystems = {
    # Upstream only ships a `linux-x86_64` binary release; aarch64 would
    # need a from-source build (Haskell, big tail of effort).
    futhark = [ "x86_64-linux" ];
    # The leanblas C extension lake-builds against `<cblas.h>` and fails
    # to link on aarch64-linux (`cblas_daxpby`/`cblas_zdot` undefined --
    # nixpkgs's OpenBLAS on aarch64 doesn't expose the same symbol set
    # OpenBLAS-on-x86_64 does). Real fix requires upstream leanblas
    # adjustments; gate for now.
    scilean = [ "x86_64-linux" ];
  };
  supportedTools = lib.filterAttrs
    (name: _: !(toolSystems ? ${name}) || lib.elem system toolSystems.${name})
    tools;

  prefix = p: set:
    lib.mapAttrs' (name: drv: lib.nameValuePair "${p}${name}" drv) set;
  imagesOf = p: set:
    lib.mapAttrs'
    (name: drv: lib.nameValuePair "image-${p}${name}" (gblib.mkImage drv)) set;
in {
  inherit gblib;
  packages = (prefix "eval-" evals) // (prefix "tool-" supportedTools)
    // (imagesOf "eval-" evals) // (imagesOf "tool-" supportedTools);
}
