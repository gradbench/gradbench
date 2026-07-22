# Enzyme (nixpkgs `enzyme`). Its Makefile uses clang++ and lld, loading the
# `LLDEnzyme-<N>.so` plugin found via ENZYME_LIB. nixpkgs's `enzyme` is built
# against the default `llvmPackages`, so we pick the same set here -- the
# major version moves in lockstep with nixpkgs.
{ pkgs, gblib }:

gblib.mkCppTool {
  name = "enzyme";
  compiler = pkgs.llvmPackages.clang;
  extraInputs = [ pkgs.llvmPackages.lld pkgs.llvmPackages.openmp ];
  extraSetup = ''
    export ENZYME_LIB="${pkgs.enzyme}/lib"
  '';
}
