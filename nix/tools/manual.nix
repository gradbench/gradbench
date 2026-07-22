# The `manual` tool: hand-written C++ reference implementations. No external AD
# library, just a C++ compiler.
{ pkgs, gblib }:

gblib.mkCppTool { name = "manual"; }
