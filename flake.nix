{
  description = "GradBench Nix flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
    }:
    let
      lib = nixpkgs.lib;
      systems = [ "x86_64-linux" "aarch64-linux" ];
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      pythonSets = lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          python = pkgs.python311;
          cudaLibs = with pkgs.cudaPackages; [
            cuda_cudart
            cuda_cupti
            cuda_nvrtc
            libcublas
            libcublasmp
            cudnn
            libcufft
            libcurand
            libcusolver
            libcusolvermp
            libcusparse
            libcusparse_lt
            libnvjitlink
          ];
          pyprojectOverrides = final: prev: {
            torch = prev.torch.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
              autoPatchelfIgnoreMissingDeps = (old.autoPatchelfIgnoreMissingDeps or [ ]) ++ [
                "libcuda.so.1"
              ];
            });
            tensorflow-io-gcs-filesystem = prev.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ prev.tensorflow ];
              postFixup = (old.postFixup or "") + ''
                addAutoPatchelfSearchPath ${prev.tensorflow}/lib/python3.11/site-packages/tensorflow
              '';
            });
            nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
            });
            nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
            });
          };
        in
        (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
          (lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            overlay
            pyprojectOverrides
          ])
      );
      evalNames =
        let
          entries = builtins.readDir ./evals;
          names = builtins.attrNames entries;
        in
        lib.filter (name:
          entries.${name} == "directory"
          && builtins.pathExists (./evals + "/${name}/default.nix")
        ) names;
      toolNames =
        let
          entries = builtins.readDir ./tools;
          names = builtins.attrNames entries;
        in
        lib.filter (name:
          entries.${name} == "directory"
          && builtins.pathExists (./tools + "/${name}/default.nix")
        ) names;
    in
    {
      packages = lib.genAttrs systems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
          src = self;
          pythonSet = pythonSets.${system};
          evalPkgs = builtins.listToAttrs (map (name: {
            name = "eval-${name}";
            value = import ./evals/${name}/default.nix { inherit pkgs src pythonSet; };
          }) evalNames);
          toolPkgs = builtins.listToAttrs (map (name: {
            name = "tool-${name}";
            value = import ./tools/${name}/default.nix { inherit pkgs src pythonSet; };
          }) toolNames);
        in
        evalPkgs // toolPkgs
      );
    };
}
