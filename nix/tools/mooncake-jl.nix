# mooncake-jl (Julia). See mkJuliaTool.
{ pkgs, gblib }:

gblib.mkJuliaTool {
  name = "mooncake-jl";
  depotHash = "sha256-n2YSP6zHI0DVIeroq/KDCLxZvgsuZSUuRrwIxHrCNts=";
}
