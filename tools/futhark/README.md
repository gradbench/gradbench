# Futhark

[Futhark][] is a purely functional array language with automatic
differentiation built in.

The tool accepts a `--backend` option to change which Futhark compiler
backend to use. Unless you also modify the Dockerfile to enable GPU
passthrough, only the `c` (the default) and `multicore` backends are
likely to work.

[futhark]: https://futhark-lang.org/
