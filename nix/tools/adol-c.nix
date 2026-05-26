# ADOL-C (nixpkgs `adolc`). The Makefile links `-ladolc`.
{ pkgs, gblib }:

gblib.mkCppTool {
  name = "adol-c";
  libs = [ pkgs.adolc ];
}
