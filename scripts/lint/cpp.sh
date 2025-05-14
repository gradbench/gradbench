#!/bin/sh

which clang-format

clang-format --version

git ls-files '*.c' '*.cpp' '*.h' '*.hpp' | xargs clang-format --dry-run -Werror
