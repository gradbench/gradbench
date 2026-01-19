{ pkgs, src, pythonSet }:
let
  common = import ../../nix/common.nix { inherit pkgs src pythonSet; };
  ad_hpp = pkgs.fetchurl {
    url = "https://gitlab.stce.rwth-aachen.de/stce/ad/-/raw/v1.7.1/include/ad.hpp";
    sha256 = "0z1i9za6j6shjq162ncb86l7ivvrh3f1zp2qxs6hkw9lkc5m4qhh";
  };
in
common.mkCppTool {
  name = "ad-hpp";
  prepare = ''
    mkdir -p "$WORKDIR/tools/ad-hpp/include"
    cp ${ad_hpp} "$WORKDIR/tools/ad-hpp/include/ad.hpp"
    mkdir -p "$WORKDIR/.bin"
    cat > "$WORKDIR/.bin/wget" <<'EOF_ADHPP'
    #!/bin/sh
    set -e
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "-O" ]; then
        shift
        out="$1"
      fi
      shift
    done
    if [ -z "$out" ]; then
      echo "wget stub: missing -O" >&2
      exit 1
    fi
    rm -f "$out"
    cp ${ad_hpp} "$out"
    chmod u+w "$out"
    EOF_ADHPP
    chmod +x "$WORKDIR/.bin/wget"
    export PATH="$WORKDIR/.bin:$PATH"
  '';
}
