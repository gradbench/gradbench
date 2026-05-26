# Adept (nixpkgs `adept`). The Makefile links `-ladept`.
{ pkgs, gblib }:

gblib.mkCppTool {
  name = "adept";
  libs = [ pkgs.adept ];
}
