{ rustPlatform, fetchFromGitHub }:

rustPlatform.buildRustPackage rec {
  pname = "floretta";
  version = "0.5.0";

  src = fetchFromGitHub {
    owner = "samestep";
    repo = "floretta";
    tag = "v${version}";
    hash = "sha256-1FwcQB4h60ZO30+yx+ILVephmXj0Z/eV8S2Bv8qVTI0=";
  };

  useFetchCargoVendor = true;
  cargoHash = "sha256-VcgCgCRsMCgHDmSv2Dx4DnyUokZCq+zNUzDudr8L788=";
}
