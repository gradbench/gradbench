#!/usr/bin/env bash
set -euo pipefail
json=$(cargo run)
echo "$json" | docker run --interactive --rm "ghcr.io/gradbench/$1"
