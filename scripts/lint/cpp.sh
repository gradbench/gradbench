#!/bin/sh

which clang-format

/usr/bin/clang-format --version
/usr/bin/clang-format-19 --version
file /usr/bin/clang-format
file /usr/bin/clang-format-19

git ls-files '*.c' '*.cpp' '*.h' '*.hpp' | xargs clang-format --dry-run -Werror
