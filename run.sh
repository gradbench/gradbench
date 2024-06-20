#!/usr/bin/env bash
set -euo pipefail
docker run --interactive "ghcr.io/gradbench/$1" < gradbench.json
