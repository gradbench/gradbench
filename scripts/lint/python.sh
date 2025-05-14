#!/bin/sh

# (Some?) NixOS systems cannot run the ruff installed by uv, unless we fake an
# FHS environment with steam-run.

if which steam-run >/dev/null; then
    ruff="steam-run ruff"
else
    ruff=ruff
fi

uv run $ruff check
uv run $ruff format --check
