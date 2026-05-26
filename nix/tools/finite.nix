# The `finite` tool: finite differences. No external AD library; just a C++
# compiler (with OpenMP for its multithreaded variant, provided by gcc).
{ pkgs, gblib }:

gblib.mkCppTool { name = "finite"; }
