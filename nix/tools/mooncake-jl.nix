# mooncake-jl (Julia). See mkJuliaTool.
{ pkgs, gblib }:

gblib.mkJuliaTool {
  name = "mooncake-jl";
  depotHashes = {
    x86_64-linux = "sha256-n2YSP6zHI0DVIeroq/KDCLxZvgsuZSUuRrwIxHrCNts=";
    aarch64-linux = "sha256-Aoc1C5BXKfUNxBpmOirhLezt2gh12sLUwqrRIeqkS5Q=";
  };
}
