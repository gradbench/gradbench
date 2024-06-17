#!/usr/bin/env bash
set -euo pipefail
docker build "tools/$1" --tag "ghcr.io/gradbench/$1"
