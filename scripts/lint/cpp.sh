#!/bin/sh

git ls-files '*.c' '*.cpp' '*.h' '*.hpp' | xargs clang-format --dry-run -Werror
