# reversediff-jl (Julia). See mkJuliaTool.
{ pkgs, gblib }:

gblib.mkJuliaTool {
  name = "reversediff-jl";
  depotHashes = {
    x86_64-linux = "sha256-hx+2dUs37vIZwuG+tjE9w7pIJ8+fIrGD8X8Wgq4brMQ=";
    aarch64-linux = "sha256-Sgch14xjHyqIp815GgMOU1pAVFQSVN2Vu/JVtNELHck=";
  };
}
