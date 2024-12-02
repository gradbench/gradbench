# Futhark

[Futhark][] is a purely functional array language with automatic differentiation built in.

The tool accepts a `--backend` option to change which Futhark compiler backend to use. Unless you also modify the Dockerfile to enable GPU passthrough, only the `c` (the default) and `multicore` backends are likely to work.

To run this outside Docker, you'll first need to install Futhark and run the following command in the same directory as this `README.md` file:

```sh
futhark pkg sync
```

[futhark]: https://futhark-lang.org/
