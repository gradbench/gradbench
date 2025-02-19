# Haskell

[Haskell][] is a lazy purely functional language, with AD support
provided by the [`ad`][] package.

To run this outside Docker, you need to install `ghc` and `cabal`. You
are advised to `cabal build` it before using `cabal run` to run it
(see the command in the Dockerfile), as otherwise the build output
from `cabal` will interfere with the protocol.

[haskell]: https://haskell.org/
[ad]: https://hackage.haskell.org/package/ad
