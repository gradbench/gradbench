# The `manual` tool: hand-written C++ reference implementations. It is a
# representative C++-based tool. cpp.py implements the GradBench protocol and
# compiles `tools/manual/bin/<eval>` ON DEMAND (per `define` message), so the
# C++ toolchain must be available at run time. Compile-on-demand is a feature:
# it lets GradBench measure each tool's compilation cost.
{ pkgs, gblib }:

gblib.mkTool {
  name = "manual";
  # python3 (cpp.py uses only the standard library), make, and a C++ compiler.
  runtimeInputs = [ pkgs.python3 pkgs.gnumake pkgs.gcc ];
  setup = ''
    # Provide json.hpp where cpp/Makefile would have `wget`-ed it.
    export CPATH="${gblib.jsonInclude}/include''${CPATH:+:$CPATH}"
    # The Nix cc-wrapper disables -march=native for build purity. manual is
    # compiled immediately before running (common.mk defaults NATIVE=yes and is
    # not baked into any image), so re-enable native codegen.
    export NIX_ENFORCE_NO_NATIVE=0
  '';
  entrypoint = "python3 python/gradbench/gradbench/cpp.py manual";
}
