# Zygote (Julia). See mkJuliaTool for how the depot is built.
{ pkgs, gblib }:

gblib.mkJuliaTool {
  name = "zygote";
  depotHash = "sha256-AE8FoEYk4ZuYG93v4kzBajJlXiBV1hRrgphYkdKCKhs=";
}
