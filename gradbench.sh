#!/usr/bin/env bash
set -e
cargo build --all
target/debug/gradbench "$@"
