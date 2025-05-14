#!/bin/sh

clangformat=clang-format

# Workaround for GitHub Actions.
if [ -f /usr/bin/clang-format-19 ]; then
    clangformat=/usr/bin/clang-format-19
fi

git ls-files '*.c' '*.cpp' '*.h' '*.hpp' | xargs $clangformat --dry-run -Werror
