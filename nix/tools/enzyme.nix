# Enzyme (nixpkgs `enzyme`), built against LLVM 19. Its Makefile uses clang++
# and lld, loading the `LLDEnzyme-19.so` plugin found via ENZYME_LIB.
{ pkgs, gblib }:

gblib.mkCppTool {
  name = "enzyme";
  compiler = pkgs.llvmPackages_19.clang;
  extraInputs = [ pkgs.llvmPackages_19.lld pkgs.llvmPackages_19.openmp ];
  extraSetup = ''
    export ENZYME_LIB="${pkgs.enzyme}/lib"
  '';
}
