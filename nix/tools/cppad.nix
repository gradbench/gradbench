# CppAD (nixpkgs `cppad`). Its Makefile links via `pkg-config --libs cppad`, so
# pkg-config is on PATH and cppad's lib/pkgconfig is on PKG_CONFIG_PATH.
{ pkgs, gblib }:

gblib.mkCppTool {
  name = "cppad";
  libs = [ pkgs.cppad ];
  extraInputs = [ pkgs.pkg-config ];
}
