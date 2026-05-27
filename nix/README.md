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

**34 of 36 images converted**, each verified by running it against its evals:

- **All 12 evals** (Nixpkgs Python).
- **C++ tools** (compile-on-demand): manual, finite, codipack, cppad, adol-c,
  adept, enzyme, ad-hpp.
- **JavaScript**: floretta, tensorflow-js.
- **OCaml**: ocaml.
- **Python/ML** (uv2nix, CPU): pytorch, jax, tensorflow, futhark, tapenade.
- **Julia** (fixed-output depots): zygote, forwarddiff-jl, reversediff-jl,
  mooncake-jl, enzyme-jl.
- **Haskell**: haskell-ad (callCabal2nix, GHC 9.12).

Not yet converted (2), both blocked on toolchains absent from the pinned
Nixpkgs — see below.

## Known follow-ups

These are tracked for later phases of the migration:

- **horde-ad** (Haskell): needs **GHC 9.14** (pinned Nixpkgs has only 9.10 and
  9.12) and the `horde-ad`/`ox-arrays` libraries, which aren't in Nixpkgs, at a
  pinned Hackage `index-state`. Realistic options: adopt `haskell.nix` (handles
  `cabal.project`, `index-state`, `--allow-newer`, and its own GHCs natively),
  or bump the Nixpkgs pin to one with GHC 9.14 and package the missing
  libraries. This is an architectural decision (extra flake input or pin bump).
- **scilean** (Lean): the `lean-toolchain` pins **Lean v4.16.0** (Nixpkgs has
  4.19.0), and it pulls SciLean from a git branch plus prebuilt `.olean` caches
  via `lake exe cache get`. Likely approach: a fixed-output derivation that uses
  `elan` (in Nixpkgs) to fetch the pinned toolchain and runs `lake exe cache
  get` + `lake build`, then autoPatchelf the resulting binary and shared libs.
  Determinism of the Lake build/cache is the risk.
- **CI**: workflows still build Docker images. The plan is to pass built
  derivations between jobs as serialized store closures
  (`nix-store --export` → artifact → `nix-store --import`); the CLI's
  `repo run --download-github` path already expects `.closure` artifacts.
- **`nix run` startup**: `gradbench repo run` spawns two `nix run` processes
  concurrently, which can print a harmless "SQLite database is busy" eval-cache
  warning. A cleaner approach is to have the run command exec the already-built
  store path directly instead of re-evaluating via `nix run`.
- **Multi-platform**: the flake currently targets `x86_64-linux` only; extend
  `systems` in `flake.nix` for `aarch64-linux` / `aarch64-darwin`.
- **Cleanup**: once parity is reached, remove the `Dockerfile`s,
  `.dockerignore`, the legacy `shell.nix`, and switch `.envrc` to `use flake`.
