# CoDiPack: a header-only C++ AD library (nixpkgs `codipack`).
{ pkgs, gblib }:

gblib.mkCppTool {
  name = "codipack";
  libs = [ pkgs.codipack ];
}
