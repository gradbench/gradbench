#!/usr/bin/env bash
set -euo pipefail
docker run --rm --interactive "ghcr.io/gradbench/tool-$1"
