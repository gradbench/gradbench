# Horde-ad

[Haskell][] is a lazy purely functional language, with AD support provided by
the [`horde-ad`][] package.

To run this outside Docker, you need to install `ghc` and `cabal`. You are
advised to `cabal build` it before using `cabal run` to run it (see the command
in the Dockerfile), as otherwise the build output from `cabal` will interfere
with the protocol.

[`horde-ad`]: https://hackage.haskell.org/package/horde-ad
[haskell]: https://haskell.org/

## Running outside of Docker

You need a Haskell development environment, including `cabal` and `ghc`. The
easiest way to get one is to use [ghcup][] or [shell.nix][].

Then run the following command to populate local knowledge of the Hackage
package repository:

```
$ cabal update
```

Use this command to compile the tool:

```
$ cabal build --project-dir tools/horde-ad gradbench
```

And finally this command to run it:

```
$ cabal run --project-dir tools/horde-ad gradbench
```

The `cabal run` command will actually compile if necessary as well, but its
diagnostic output can interfere with the protocol when used as the `--tool`
argument to `gradbench`.

[ghcup]: https://www.haskell.org/ghcup/
[shell.nix]: /shell.nix
