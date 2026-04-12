# Horde-ad

[`horde-ad`][] is an AD library for [Haskell][], the lazy purely functional
programming language.

To run this outside Docker, you need to install `ghc` and `cabal`.
The easiest way to get these is to use [ghcup][] or [shell.nix][].

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

[`horde-ad`]: https://hackage.haskell.org/package/horde-ad
[haskell]: https://haskell.org/
[ghcup]: https://www.haskell.org/ghcup/
[shell.nix]: /shell.nix
