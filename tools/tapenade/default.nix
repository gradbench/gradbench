{ pkgs, src, pythonSet }:
let
  common = import ../../nix/common.nix { inherit pkgs src pythonSet; };
  tapenadeTar = pkgs.fetchurl {
    url = "https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar";
    sha256 = "00x74fzj1gl4cnxym51z28i7npf2ba8w2kx8hhpc5i7x4kqmr95k";
  };
  tapenade = pkgs.stdenv.mkDerivation {
    name = "tapenade-3.16";
    dontUnpack = true;
    installPhase = ''
      mkdir -p $out
      tar -xf ${tapenadeTar} -C $out
    '';
  };
in
common.mkPythonTool {
  name = "tapenade";
  extraInputs = [ pkgs.gcc pkgs.gnumake pkgs.openjdk ];
  groups = [ "scipy" ];
  useWritableRepo = true;
  prepare = ''
    cp -r ${tapenade}/tapenade_3.16 "$WORKDIR/"
  '';
}
