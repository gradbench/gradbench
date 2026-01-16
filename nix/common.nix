{ pkgs ? let sources = import ./sources.nix; in import sources.nixpkgs { }
, src ? ../.
, pythonSet ? null
}:
let
  lib = pkgs.lib;
  jsonHpp = pkgs.fetchurl {
    url = "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp";
    sha256 = "19iy237x05xl41fsz1v0viznz4pq09is7r9bdch1qjpgcs04rslv";
  };

  mkRunner =
    { name
    , kind
    , command
    , runtimeInputs ? [ ]
    , env ? { }
    , workdir ? "${src}"
    , prepare ? ""
    }:
    pkgs.stdenv.mkDerivation {
      name = "gradbench-${kind}-${name}";
      dontUnpack = true;
      installPhase = ''
        mkdir -p $out/bin
        cat > $out/bin/run <<'EOF'
        #!${pkgs.bash}/bin/bash
        set -euo pipefail
        export PATH=${lib.makeBinPath runtimeInputs}:$PATH
        ${lib.concatStringsSep "\n" (map (k: "export ${k}=${lib.escapeShellArg env.${k}}") (lib.attrNames env))}
        WORKDIR=${lib.escapeShellArg workdir}
        ${prepare}
        cd "$WORKDIR"
        exec ${command} "$@"
        EOF
        chmod +x $out/bin/run
      '';
    };

  prepareWritableRepo = ''
    tmpdir=$(mktemp -d)
    cp -r ${src} "$tmpdir/gradbench"
    chmod -R u+w "$tmpdir/gradbench"
    mkdir -p "$tmpdir/gradbench/cpp"
    cp ${jsonHpp} "$tmpdir/gradbench/cpp/json.hpp"
    WORKDIR="$tmpdir/gradbench"
  '';

  basePythonGroups = [
    "numpy"
    "pydantic"
    "dataclasses-json"
  ];

  evalPythonGroups = basePythonGroups ++ [ "scipy" ];

  mkUvPythonEnv =
    { name
    , groups
    }:
    if pythonSet == null then
      throw "pythonSet is required to build Python environments from uv.lock"
    else
      pythonSet.mkVirtualEnv "gradbench-${name}-env" {
        gradbench-workspace = groups;
      };

  pythonEvalEnv = mkUvPythonEnv {
    name = "eval";
    groups = evalPythonGroups;
  };

  mkEval = { name }:
    mkRunner {
      inherit name;
      kind = "eval";
      command = "${pythonEvalEnv}/bin/python ${src}/python/gradbench/gradbench/evals/${name}/run.py";
      runtimeInputs = [
        pythonEvalEnv
        pkgs.gnumake
        pkgs.gcc
        pkgs.pkg-config
        pkgs.eigen
        pkgs.openblas
        pkgs.lapack
        pkgs.blas
        pkgs.zlib
      ];
      env = {
        PYTHONPATH = "${src}/python/gradbench";
      };
      prepare = prepareWritableRepo + ''
        if [ -f tools/manual/evals.txt ] && grep -qx "${name}" tools/manual/evals.txt; then
          make -C tools/manual bin/${name} -B MULTITHREADED=no >/dev/null 2>&1 || true
        fi
      '';
    };

  mkPythonTool =
    { name
    , extraInputs ? [ ]
    , scriptPath ? "python/gradbench/gradbench/tools/${name}/run.py"
    , prepare ? ""
    , groups ? [ ]
    , useWritableRepo ? false
    }:
    let
      pythonEnv = mkUvPythonEnv {
        name = "tool-${name}";
        groups = basePythonGroups ++ groups;
      };
    in
    mkRunner {
      inherit name;
      kind = "tool";
      command = "${pythonEnv}/bin/python ${src}/${scriptPath}";
      runtimeInputs = [ pythonEnv ] ++ extraInputs;
      env = {
        PYTHONPATH = "${src}/python/gradbench";
      };
      prepare = (if useWritableRepo then prepareWritableRepo else "") + prepare;
    };

  mkCppTool =
    { name
    , extraInputs ? [ ]
    , prepare ? ""
    }:
    mkRunner {
      inherit name;
      kind = "tool";
      command = "${pkgs.python311}/bin/python ${src}/python/gradbench/gradbench/cpp.py ${name}";
      runtimeInputs =
        [
          pkgs.gnumake
          pkgs.gcc
          pkgs.pkg-config
          pkgs.eigen
          pkgs.openblas
          pkgs.lapack
          pkgs.blas
          pkgs.zlib
        ]
        ++ extraInputs;
      prepare = prepareWritableRepo + prepare;
    };

  mkJuliaTool =
    { name }:
    let
      julia = pkgs.julia_110-bin;
      juliaCache = "\${XDG_CACHE_HOME:-$HOME/.cache}/gradbench/julia/${name}";
    in
    mkRunner {
      inherit name;
      kind = "tool";
      command = "${julia}/bin/julia --project=${src}/tools/${name} ${src}/tools/${name}/run.jl";
      runtimeInputs = [ julia ];
      prepare = ''
        mkdir -p ${juliaCache}
        export JULIA_DEPOT_PATH=${juliaCache}
        export JULIA_PKG_PRECOMPILE_AUTO=0
        if [ ! -f ${juliaCache}/.instantiated ]; then
          ${julia}/bin/julia --project=${src}/tools/${name} -e 'using Pkg; Pkg.instantiate()'
          touch ${juliaCache}/.instantiated
        fi
      '';
    };

  mkNodeTool =
    { name
    , scriptPath
    , extraInputs ? [ ]
    , prepare ? ""
    }:
    mkRunner {
      inherit name;
      kind = "tool";
      command = "${pkgs.nodejs_24}/bin/node --disable-warning=ExperimentalWarning ${scriptPath}";
      runtimeInputs = [ pkgs.nodejs_24 ] ++ extraInputs;
      inherit prepare;
    };

  mkHaskellTool = { name }:
    let
      haskellTool = pkgs.haskellPackages.callCabal2nix "gradbench-haskell" (src + "/tools/haskell") { };
    in
    mkRunner {
      inherit name;
      kind = "tool";
      command = "${haskellTool}/bin/gradbench";
    };

  mkOcamlTool = { name }:
    let
      ocamlPackages = pkgs.ocamlPackages.overrideScope (self: super: {
        owl = super.owl.overrideAttrs (old: {
          patches = (old.patches or [ ]) ++ [ ../nix/patches/ocaml-owl-exponpow.patch ];
        });
      });
      ocamlTool = ocamlPackages.buildDunePackage {
        pname = "gradbench";
        version = "0.0.0";
        src = src + "/tools/ocaml";
        propagatedBuildInputs = [ ocamlPackages.owl ocamlPackages.yojson ];
      };
    in
    mkRunner {
      inherit name;
      kind = "tool";
      command = "${ocamlTool}/bin/gradbench";
    };

  mkScileanTool = { name }:
    let
      elanCache = "\${XDG_CACHE_HOME:-$HOME/.cache}/gradbench/elan";
      lakeCache = "\${XDG_CACHE_HOME:-$HOME/.cache}/gradbench/lake";
      scileanCache = "\${XDG_CACHE_HOME:-$HOME/.cache}/gradbench/scilean";
    in
    mkRunner {
      inherit name;
      kind = "tool";
      command = "tools/scilean/.lake/build/bin/gradbench";
      runtimeInputs = [ pkgs.elan pkgs.gnumake pkgs.gcc pkgs.binutils pkgs.openblas pkgs.gawk pkgs.rsync ];
      prepare = ''
        cache_root=${scileanCache}
        repo_cache="$cache_root/repo"
        mkdir -p "$cache_root"
        if [ ! -d "$repo_cache" ]; then
          ${pkgs.rsync}/bin/rsync -a \
            --exclude '.git' \
            --exclude '.lake' \
            --exclude '.venv' \
            ${src}/ "$repo_cache/"
        else
          ${pkgs.rsync}/bin/rsync -a --delete \
            --exclude '.git' \
            --exclude '.lake' \
            --exclude '.venv' \
            ${src}/ "$repo_cache/"
        fi
        chmod -R u+w "$repo_cache"
        mkdir -p "$repo_cache/cpp"
        cp ${jsonHpp} "$repo_cache/cpp/json.hpp"
        WORKDIR="$repo_cache"
        export CPATH=${pkgs.openblas}/include${"$"}{CPATH:+":${"$"}CPATH"}
        export LIBRARY_PATH=${pkgs.openblas}/lib${"$"}{LIBRARY_PATH:+":${"$"}LIBRARY_PATH"}
        export ELAN_HOME=${elanCache}
        export LAKE_HOME=${lakeCache}
        export PATH="$ELAN_HOME/bin:$PATH"
        toolchain="$(${pkgs.coreutils}/bin/cat "$WORKDIR/tools/scilean/lean-toolchain")"
        if ! ${pkgs.elan}/bin/elan toolchain list | ${pkgs.gnugrep}/bin/grep -q "^$toolchain"; then
          ${pkgs.elan}/bin/elan toolchain install "$toolchain"
        fi
        ${pkgs.elan}/bin/elan default "$toolchain"
        ${pkgs.elan}/bin/lake -d "$WORKDIR/tools/scilean" update 1>&2
        leanblas_levelone="$WORKDIR/tools/scilean/.lake/packages/leanblas/c/levelone.c"
        if [ -f "$leanblas_levelone" ] && ! ${pkgs.gnugrep}/bin/grep -q "leanblas_cblas_daxpby_fallback" "$leanblas_levelone"; then
          ${pkgs.gawk}/bin/awk '
            { print }
            $0 == "#include \"util.h\"" {
              print ""
              print "#ifndef HAVE_CBLAS_DAXPBY"
              print "static void leanblas_cblas_daxpby_fallback(const int n, const double alpha, const double *x, const int incx,"
              print "                                           const double beta, double *y, const int incy){"
              print "  for (int i = 0; i < n; i++){"
              print "    y[i * incy] = alpha * x[i * incx] + beta * y[i * incy];"
              print "  }"
              print "}"
              print "#define cblas_daxpby leanblas_cblas_daxpby_fallback"
              print "#endif"
              print ""
            }
          ' "$leanblas_levelone" > "$leanblas_levelone.tmp"
          ${pkgs.coreutils}/bin/mv "$leanblas_levelone.tmp" "$leanblas_levelone"
        fi
        ${pkgs.elan}/bin/lake -d "$WORKDIR/tools/scilean" build 1>&2
      '';
    };

  mkFutharkTool = { name }:
    mkPythonTool {
      inherit name;
      scriptPath = "tools/futhark/run.py";
      groups = [ "futhark-server" ];
      extraInputs = [ pkgs.futhark ];
      prepare = prepareWritableRepo;
    };

  mkFlorettaTool = { name }:
    mkNodeTool {
      inherit name;
      scriptPath = "js/floretta/run.ts";
      extraInputs = [ pkgs.wasm-tools (pkgs.callPackage ./floretta.nix { }) ];
      prepare = prepareWritableRepo + ''
        mkdir -p "$WORKDIR/node_modules/@gradbench"
        ln -sfn "$WORKDIR/js/common" "$WORKDIR/node_modules/@gradbench/common"
      '';
    };

  mkTensorflowJsTool = { name }:
    mkNodeTool {
      inherit name;
      scriptPath = "js/tensorflow/run.ts";
      prepare = prepareWritableRepo + ''
        mkdir -p "$WORKDIR/node_modules/@gradbench"
        ln -sfn "$WORKDIR/js/common" "$WORKDIR/node_modules/@gradbench/common"
      '';
    };
in
{
  inherit pkgs;
  inherit
    mkEval
    mkPythonTool
    mkCppTool
    mkJuliaTool
    mkNodeTool
    mkHaskellTool
    mkOcamlTool
    mkScileanTool
    mkFutharkTool
    mkFlorettaTool
    mkTensorflowJsTool
    ;
}
