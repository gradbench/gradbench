#!/usr/bin/env bash
set -euo pipefail
docker build --platform linux/arm64 "tools/$1" --tag "ghcr.io/gradbench/$1"
