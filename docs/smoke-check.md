# Smoke check (local)

This runs every tool against every eval once per input, with reduced input sizes to keep the run short.

## Prereqs

- Enter the dev shell from the repo root:

```sh
nix-shell
```

## Run

```sh
./gradbench repo run --timeout 900 -o run-smoke \
  --eval hello \
  --eval "ba --min 1 --max 1 --min-runs 1 --min-seconds 0" \
  --eval "det -l 2 --min-runs 1 --min-seconds 0" \
  --eval "gmm -d 2 -k 2 -n 10 --min-runs 1 --min-seconds 0" \
  --eval "ht --min 1 --max 1 --model small --variant simple --min-runs 1 --min-seconds 0" \
  --eval "kmeans -k 2 -n 10 -d 2 --min-runs 1 --min-seconds 0" \
  --eval "llsq -n 16 -m 4 --min-runs 1 --min-seconds 0" \
  --eval "lse -n 100 --min-runs 1 --min-seconds 0" \
  --eval "lstm -l 1 -c 64 --min-runs 1 --min-seconds 0" \
  --eval "ode -n 10 -s 1 --min-runs 1 --min-seconds 0" \
  --eval "particle --min-runs 1 --min-seconds 0" \
  --eval "saddle --min-runs 1 --min-seconds 0"
```

The logs land in `run-smoke/<eval>/<tool>.jsonl`.

## Troubleshooting

- If a specific eval needs even smaller inputs, run it directly and check its `--help` output for flags:

```sh
./gradbench repo eval gmm -- --help
```
