# ad-hpp (RWTH Aachen STCE's `ad.hpp`), a single header normally downloaded by
# the Makefile. We fetch it reproducibly and materialize it where the Makefile
# expects (tools/ad-hpp/include/ad.hpp), so no network access is needed at build
# or run time and the `include/ad.hpp` make dependency is already satisfied.
{ pkgs, gblib }:

let
  adHpp = pkgs.fetchurl {
    url = "https://gitlab.stce.rwth-aachen.de/stce/ad/-/raw/v1.7.1/include/ad.hpp";
    hash = "sha256-EGJSC5s08QmN7ljcH9yAee94qEGLWWECllAbadRPMXw=";
  };
in gblib.mkCppTool {
  name = "ad-hpp";
  extraInputs = [ pkgs.coreutils ];
  extraSetup = ''
    mkdir -p "$root/tools/ad-hpp/include"
    install -m 0644 ${adHpp} "$root/tools/ad-hpp/include/ad.hpp"
  '';
}
