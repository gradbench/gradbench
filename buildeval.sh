#!/usr/bin/env bash
set -euo pipefail
docker build . --file "evals/$1/Dockerfile" --tag "ghcr.io/gradbench/eval-$1"
