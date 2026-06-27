# Zygote (Julia). See mkJuliaTool for how the depot is built.
{ pkgs, gblib }:

gblib.mkJuliaTool {
  name = "zygote";
  depotHashes = {
    x86_64-linux = "sha256-AE8FoEYk4ZuYG93v4kzBajJlXiBV1hRrgphYkdKCKhs=";
    aarch64-linux = "sha256-DuP7OvHr8Pvbpc/YevQVoW58dRNTuCmsfbSuDPerbnk=";
  };
}
