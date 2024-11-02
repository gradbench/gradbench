#!/usr/bin/env bash
set -euo pipefail
jq -cs '{golden: .[0], log: .[1]}' golden/log.json log.json | docker run --rm --interactive --entrypoint /home/gradbench/validate "ghcr.io/gradbench/eval-$1"
