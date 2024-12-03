#!/usr/bin/env bash
set -euo pipefail
eval=$1
shift
docker run --rm --interactive "ghcr.io/gradbench/eval-${eval}" "$@"
