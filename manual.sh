#!/usr/bin/env bash
set -euo pipefail
docker build --platform linux/amd64,linux/arm64 "docker/$1" --tag "ghcr.io/gradbench/$1"
tag=$(docker run --rm "ghcr.io/gradbench/$1")
docker tag "ghcr.io/gradbench/$1" "ghcr.io/gradbench/$1:$tag"
docker push "ghcr.io/gradbench/$1:$tag"
