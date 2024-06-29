#!/usr/bin/env bash
set -euo pipefail
docker build . --file "tools/$1/Dockerfile" --tag "ghcr.io/gradbench/tool-$1"
