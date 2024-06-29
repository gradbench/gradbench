#!/usr/bin/env bash
set -euo pipefail
docker build --platform linux/amd64,linux/arm64 . --file "tools/$1/Dockerfile" --tag "ghcr.io/gradbench/tool-$1"
