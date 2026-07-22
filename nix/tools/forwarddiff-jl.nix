# forwarddiff-jl (Julia). See mkJuliaTool.
{ pkgs, gblib }:

gblib.mkJuliaTool {
  name = "forwarddiff-jl";
  depotHashes = {
    x86_64-linux = "sha256-7ndj3SGKpIw7/9WkMNAc08IgsA0uxap4vHIX0LOHJRk=";
    aarch64-linux = "sha256-ZFdY31dl9F5gjm/ft5L4ZJj6MP5/QGecKkycO1dSrBk=";
  };
}
