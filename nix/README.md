# Nix packaging

GradBench builds and runs each eval and tool as a **Nix derivation** (a wrapper
script that bakes in the dependencies and entrypoint that used to live in a
per-eval/per-tool `Dockerfile`), which can either be run directly on the host or
wrapped into an OCI image via `dockerTools`.

This directory and the top-level [`flake.nix`](../flake.nix) implement that. All
36 evals and tools are converted (the old `Dockerfile`s have been removed).

## Layout

- [`../flake.nix`](../flake.nix) — inputs (pinned Nixpkgs) and the per-system
  outputs (`packages`, `apps`, `devShells`, `formatter`).
- [`registry.nix`](registry.nix) — the list of converted evals and tools. **This
  is the file you edit to add an eval or tool.** Everything else (native
  runners, OCI images, `nix run` apps) is derived from it.
- [`lib.nix`](lib.nix) — the helpers `mkEval` / `mkTool` (native runners) and
  `mkImage` (OCI images), plus shared bits like the `json.hpp` include.
- [`evals/`](evals) and [`tools/`](tools) — one file per converted eval/tool.
- [`devshell.nix`](devshell.nix) — the development shell (`nix develop`, or
  `direnv allow` via the flake-based `.envrc`).
- [`floretta.nix`](floretta.nix) — a custom package not yet in Nixpkgs.

## Outputs

For an eval or tool named `<name>`:

| Output                                                      | What it is                                                |
| ----------------------------------------------------------- | --------------------------------------------------------- |
| `packages.<system>.eval-<name>` / `tool-<name>`             | native runner; `nix build` it or run via the app          |
| `apps.<system>.eval-<name>` / `tool-<name>`                 | `nix run .#eval-<name>` / `.#tool-<name>`                 |
| `packages.<system>.image-eval-<name>` / `image-tool-<name>` | OCI image (`dockerTools`) built from the runner's closure |

## Run modes

- **Native** (`nix run .#tool-<name>`): the wrapper runs against the working
  tree. `GRADBENCH_SOURCE_ROOT` defaults to the current directory, which must be
  a GradBench checkout. This keeps compile-on-demand tools writable (they build
  into `tools/<tool>/bin`), which is intentional: GradBench measures each tool's
  compilation cost at run time.
- **OCI image**: `mkImage` embeds the source read-only and copies it to a
  writable workdir at container startup, so compile-on-demand still works.

The GradBench CLI uses these: `gradbench eval/tool <name>` runs `.#eval-<name>`
/ `.#tool-<name>`, and `gradbench repo run --eval <name> --tool <name>` builds
them with `nix build` and then runs them.

## Adding an eval or tool

1. Create `nix/evals/<name>.nix` or `nix/tools/<name>.nix`, returning
   `gblib.mkEval { ... }` / `gblib.mkTool { ... }`. Translate the old
   `Dockerfile`: `runtimeInputs` are the dependencies, `setup` holds any
   `export`s, and `entrypoint` is the `ENTRYPOINT` command.
2. Register it in [`registry.nix`](registry.nix).
3. Verify: `nix run .#tool-<name>` and
   `gradbench repo run --eval hello --tool <name>` (or an eval the tool
   supports).

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
/ `.#tool-<name>`, serialize the result's store closure (`nix-store --export` →
a `<name>.closure` artifact), and the `run` job imports them
(`gradbench repo run --download-github`, which downloads the artifacts and
`nix-store --import`s them, then `nix run`s). Nix is installed via
`.github/actions/nix`, which also enables IOG's binary cache. The `lint`/`site`
jobs are unchanged, and OCI image publishing to GHCR is dropped for now (the
`image-*` flake outputs and `dockerTools` are still there to re-enable it).

## Binary caches

We **read** from public upstream substituters but do **not** run our own
**push** cache: CI moves built closures between the `build` and `run` jobs as
artifacts (see [CI](#ci) above), the same way the old setup pushed images to
GHCR but never actually pulled them in practice. The reasoning is recorded under
[Why no push cache](#why-no-push-cache) below.

Two tools build large toolchains from source unless an upstream cache is read:

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

  The Lake _build_ output is ~35.8 GB (mathlib oleans plus the dereferenced,
  writable dependency trees SciLean's module precompilation needs). None of that
  is needed at run time: the `gradbench` executable statically links the Lean
  runtime, SciLean and mathlib and -- verified by `strace` -- opens nothing from
  the Lake tree, needing only BLAS and libc. So `scilean.nix` exposes a
  `scilean-bin` derivation that copies out just the binary and uses
  `remove-references-to` to strip the one dead `.rodata` reference to the 3.2 GB
  Lean toolchain, leaving a 397 MiB runtime closure. The full build tree is
  still produced (and would be cached by a hosted cache), but only the 397 MiB
  runtime closure is shipped (as an artifact) to the `run` jobs.

## Why no push cache

A push cache (e.g. the GitHub-native Magic Nix Cache, or a hosted one like
Garnix) would let jobs share builds instead of passing closures as artifacts. We
measured what such a cache would have to hold -- the union, deduplicated, of
every locally-built (non-upstream) store path across all 36 build jobs,
compressed (what counts against a cache's quota):

| segment                                                                                            | uncompressed | compressed |
| -------------------------------------------------------------------------------------------------- | -----------: | ---------: |
| the other 35 evals/tools                                                                           |       6.6 GB |    2.44 GB |
| scilean's dependency builds (compiled SciLean, mathlib oleans, the `lake exe cache get` FOD, lean) |        29 GB |    8.07 GB |
| scilean's final Lake output (not needed; see above)                                                |        34 GB |          — |

The 35 non-scilean images compress to **2.44 GB** -- a comfortable fit for the
**GitHub Actions cache (~10 GB/repo, LRU)** that Magic Nix Cache uses. But
scilean's _dependency_ builds alone are **~8 GB compressed**: lean4-nix realizes
each Lake dependency (SciLean, mathlib, ...) as its own store path full of
`.olean`s plus the dereferenced writable copies `makeDepsWritable` makes, and a
post-build-hook cache uploads all of them -- the runtime trim above can't avoid
that, since those are build inputs, not the final output. 8 GB + 2.44 GB busts
the 10 GB budget and would thrash it via LRU. Caching scilean's build therefore
needs a hosted cache with no 10 GB ceiling (Garnix's free-tier storage limit is
unpublished and 8 GB may exceed it; self-hosted S3/MinIO or paid Cachix would
work). For now we run no push cache at all, matching the prior GHCR behaviour;
this is the first thing to revisit if CI build time becomes a problem.

## Other follow-ups

- **`nix run` startup**: `gradbench repo run` spawns two `nix run` processes
  concurrently, which can print a harmless "SQLite database is busy" eval-cache
  warning; and for horde-ad it re-runs haskell.nix's plan IFD on each run. A
  cleaner approach is to exec the already-built store path directly instead of
  re-evaluating via `nix run`.
- **Multi-platform**: the flake currently targets `x86_64-linux` only; extend
  `systems` in `flake.nix` for `aarch64-linux` / `aarch64-darwin`.
- **Usage docs**: the `Dockerfile`s, `.dockerignore`, legacy `shell.nix` and niv
  `nix/sources.*` are removed, and `.envrc` uses the flake. The Docker-centric
  _narrative_ in the top-level [`README.md`](../README.md) and
  [`CONTRIBUTING.md`](../CONTRIBUTING.md) (the "## Docker" section, image
  building, `--platform` emulation) still needs a pass to match the Nix
  workflow; that is best done alongside finalizing the CLI surface and the
  multi-platform work above.
- **scilean build disk**: building scilean realizes ~63 GB of store paths (~34
  GB final output + ~29 GB dependency builds), which exceeds the ~14 GB free
  disk on standard GitHub-hosted runners. Its `build` job will need a
  larger/self-hosted runner, or scilean's Lake outputs would need slimming (e.g.
  avoiding `makeDepsWritable`'s `cp -rL` duplication of mathlib/batteries into
  each consumer's `.lake`).
