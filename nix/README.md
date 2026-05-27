# Nix packaging

GradBench is migrating from per-eval/per-tool `Dockerfile`s to a single Nix
flake. Each eval and tool becomes a **runnable Nix derivation** (a wrapper
script that bakes in the dependencies and entrypoint that used to live in the
`Dockerfile`), which can either be run directly on the host or wrapped into an
OCI image via `dockerTools`.

This directory and the top-level [`flake.nix`](../flake.nix) implement that.
This is a work in progress: only some evals and tools have been converted.

## Layout

- [`../flake.nix`](../flake.nix) — inputs (pinned Nixpkgs) and the per-system
  outputs (`packages`, `apps`, `devShells`, `formatter`).
- [`registry.nix`](registry.nix) — the list of converted evals and tools.
  **This is the file you edit to add an eval or tool.** Everything else (native
  runners, OCI images, `nix run` apps) is derived from it.
- [`lib.nix`](lib.nix) — the helpers `mkEval` / `mkTool` (native runners) and
  `mkImage` (OCI images), plus shared bits like the `json.hpp` include.
- [`evals/`](evals) and [`tools/`](tools) — one file per converted eval/tool.
- [`devshell.nix`](devshell.nix) — the development shell (`nix develop`); the
  flake-based successor to the legacy niv-pinned [`../shell.nix`](../shell.nix).
- [`floretta.nix`](floretta.nix) — a custom package not yet in Nixpkgs.

## Outputs

For an eval or tool named `<name>`:

| Output | What it is |
| --- | --- |
| `packages.<system>.eval-<name>` / `tool-<name>` | native runner; `nix build` it or run via the app |
| `apps.<system>.eval-<name>` / `tool-<name>` | `nix run .#eval-<name>` / `.#tool-<name>` |
| `packages.<system>.image-eval-<name>` / `image-tool-<name>` | OCI image (`dockerTools`) built from the runner's closure |

## Run modes

- **Native** (`nix run .#tool-<name>`): the wrapper runs against the working
  tree. `GRADBENCH_SOURCE_ROOT` defaults to the current directory, which must be
  a GradBench checkout. This keeps compile-on-demand tools writable (they build
  into `tools/<tool>/bin`), which is intentional: GradBench measures each tool's
  compilation cost at run time.
- **OCI image**: `mkImage` embeds the source read-only and copies it to a
  writable workdir at container startup, so compile-on-demand still works.

The GradBench CLI uses these: `gradbench eval/tool <name>` runs
`.#eval-<name>` / `.#tool-<name>`, and `gradbench repo run --eval <name> --tool
<name>` builds them with `nix build` and then runs them.

## Adding an eval or tool

1. Create `nix/evals/<name>.nix` or `nix/tools/<name>.nix`, returning
   `gblib.mkEval { ... }` / `gblib.mkTool { ... }`. Translate the old
   `Dockerfile`: `runtimeInputs` are the dependencies, `setup` holds any
   `export`s, and `entrypoint` is the `ENTRYPOINT` command.
2. Register it in [`registry.nix`](registry.nix).
3. Verify: `nix run .#tool-<name>` and `gradbench repo run --eval hello --tool
   <name>` (or an eval the tool supports).

See [`tools/manual.nix`](tools/manual.nix) (a compile-on-demand C++ tool) and
[`evals/hello.nix`](evals/hello.nix) (a Python eval) as templates.

## Status

**All 36 images are converted and verified** by running each against its evals.
35 build with only cache.nixos.org as a substituter (scilean additionally
fetches mathlib's prebuilt oleans from the Lean community cache; see below).
horde-ad builds GHC 9.14 and its dependency tree from source here (verified end
to end); IOG's binary cache makes that fast (see below).

- **All 12 evals** (Nixpkgs Python).
- **C++ tools** (compile-on-demand): manual, finite, codipack, cppad, adol-c,
  adept, enzyme, ad-hpp.
- **JavaScript**: floretta, tensorflow-js.
- **OCaml**: ocaml.
- **Python/ML** (uv2nix, CPU): pytorch, jax, tensorflow, futhark, tapenade.
- **Julia** (fixed-output depots): zygote, forwarddiff-jl, reversediff-jl,
  mooncake-jl, enzyme-jl.
- **Haskell**: haskell-ad (callCabal2nix, GHC 9.12).
- **Lean**: scilean (lean4-nix; mathlib's oleans from the Lean community cache,
  the other Lake deps from source). The compiled `gradbench` binary is fully
  self-contained, so the runtime closure is trimmed to just that binary (397 MiB
  total) rather than the ~35.8 GB Lake build tree (see below).
- **horde-ad** (Haskell): converted via `haskell.nix` (`tools/horde-ad.nix`),
  which resolves a GHC 9.14 plan from Hackage at the cabal `index-state` (with
  `allow-newer: all`, as the Dockerfile uses) -- picking 9.14-compatible
  dependency versions Nixpkgs lacks. Runs det/hello/llsq/gmm. Builds GHC 9.14
  from source unless IOG's cache is configured.

## CI

The GitHub Actions workflows (`.github/workflows/{build,nightly}.yml`) build
with Nix instead of Docker: the `eval`/`tool` jobs run `nix build .#eval-<name>`
/ `.#tool-<name>`, serialize the result's store closure
(`nix-store --export` → a `<name>.closure` artifact), and the `run` job imports
them (`gradbench repo run --download-github`, which downloads the artifacts and
`nix-store --import`s them, then `nix run`s). Nix is installed via
`.github/actions/nix`, which also enables IOG's binary cache. The `lint`/`site`
jobs are unchanged, and OCI image publishing to GHCR is dropped for now (the
`image-*` flake outputs and `dockerTools` are still there to re-enable it).

## Binary caches

Two tools build large toolchains from source unless a cache is available:

- **horde-ad** → **IOG's public cache** `https://cache.iog.io` (key
  `hydra.iohk.io:f/Ea+s+dFdN+3Y/G+FDgSq+a5NEWhJGzdjvKNGv0/EQ=`). CI enables it;
  for local builds add it to `nix.conf` as a trusted user, else GHC 9.14 is
  built from source.
- **scilean** (mathlib) → no Nix cache needed. mathlib isn't in Nixpkgs and the
  Lean community cache serves `.olean`s over HTTP (not a Nix substituter), so
  instead `tools/scilean.nix` runs `lake exe cache get` in a fixed-output
  derivation (`mathlibCache`) to fetch mathlib's prebuilt `.olean`/`.c` for the
  pinned rev (~40s), then injects them so Lake skips elaborating mathlib and
  only compiles its `.c` to `.so`. SciLean's own elaboration and that `.so`
  compilation still take a while, but mathlib -- by far the largest piece -- is
  never elaborated from source. (If even that becomes a CI bottleneck, a hosted
  cache like Garnix would cache the whole build.)

  The Lake *build* output is ~35.8 GB (mathlib oleans plus the dereferenced,
  writable dependency trees SciLean's module precompilation needs). None of that
  is needed at run time: the `gradbench` executable statically links the Lean
  runtime, SciLean and mathlib and -- verified by `strace` -- opens nothing from
  the Lake tree, needing only BLAS and libc. So `scilean.nix` exposes a
  `scilean-bin` derivation that copies out just the binary and uses
  `remove-references-to` to strip the one dead `.rodata` reference to the 3.2 GB
  Lean toolchain, leaving a 397 MiB runtime closure. The full build tree is
  still produced (and would be cached by a hosted cache), but never shipped to
  the `run` jobs.

## Other follow-ups

- **`nix run` startup**: `gradbench repo run` spawns two `nix run` processes
  concurrently, which can print a harmless "SQLite database is busy" eval-cache
  warning; and for horde-ad it re-runs haskell.nix's plan IFD on each run. A
  cleaner approach is to exec the already-built store path directly instead of
  re-evaluating via `nix run`.
- **Multi-platform**: the flake currently targets `x86_64-linux` only; extend
  `systems` in `flake.nix` for `aarch64-linux` / `aarch64-darwin`.
- **Cleanup**: once parity is reached, remove the `Dockerfile`s,
  `.dockerignore`, the legacy `shell.nix`, and switch `.envrc` to `use flake`.
