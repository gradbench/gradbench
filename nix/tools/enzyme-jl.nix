# enzyme-jl (Julia). See mkJuliaTool.
{ pkgs, gblib }:

gblib.mkJuliaTool {
  name = "enzyme-jl";
  # `Pkg.instantiate()`'s precompiled artifacts differ per platform; record
  # the hash for each system we've built on. The lib.fakeHash placeholder
  # provokes a hash-mismatch error on systems we haven't measured yet --
  # the error spells out the real hash to paste in.
  depotHashes = {
    x86_64-linux = "sha256-ek+Po0i9Bw6moKWhYM+55f4tAuUUt+XxS1p8ONjwtCc=";
    aarch64-linux = "sha256-Jt+3DzL51NmO5lemUVlTe39CiztGMK2zX4ndf1WF4rw=";
  };
}
