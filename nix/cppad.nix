{
  lib,
  stdenv,
  fetchFromGitHub,
  pkg-config,
  cmake,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "cppad";
  version = "20250000.2";

  src = fetchFromGitHub {
    owner = "coin-or";
    repo = "CppAD";
    tag = "${finalAttrs.version}";
    hash = "sha256-rAKD/PAjepDchvrJp7iLYw5doNq8Af1oVh61gfMcNYI=";
  };

  nativeBuildInputs = [
    cmake
    pkg-config
  ];

  meta = with lib; {
    description = "A C++ Algorithmic Differentiation Package";
    homepage = "https://github.com/coin-or/CppAD";
    license = licenses.gpl2Plus;
    maintainers = with maintainers; [
      athas
    ];
  };
})
