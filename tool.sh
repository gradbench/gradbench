#!/usr/bin/env bash
set -euo pipefail
tool=$1
shift
docker run --rm --interactive "ghcr.io/gradbench/tool-${tool}" "$@"
