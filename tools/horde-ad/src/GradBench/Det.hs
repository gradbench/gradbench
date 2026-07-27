{-# LANGUAGE OverloadedLists #-}
-- | This code directly implements the recursive specification
-- and so is fundamentally inefficient (something like O(n!)).
-- Fortunately, the workloads are necessarily tiny so this doesn't OOM.
-- Due to the tiny workloads and, consequently, numerous but tiny tensors
-- with their metadata overhead, the gradient, but especially the primal,
-- are slower than when implemented with lists for haskell-ad (and lists
-- are handled in horde-ad via tensors, so using them in the code below
-- would not help).
--
-- We employ symbolic @grad@ instead of non-symbolic @cgrad@, because
-- we use @rbuild1@, which takes much more memory and time with @cgrad@.
--
-- TODO: transform @rbuild1@ manually to @rgather@, manually fuse and simplify
-- the resulting code as much as possible and see if it's more performant
-- that currently (try both @grad@ and @cgrad@; the latter should be
-- equally fast when the fusion and simplification is completed manually
-- and when AD is redone for each individual run, as it is currently).
--
-- TODO2: once https://gitlab.haskell.org/ghc/ghc/-/issues/26816 is solved,
-- bring back from previous commits the implementation copied from Futhark
-- and try to make it faster than this one.
module GradBench.Det
  ( Input,
    PrimalOutput,
    GradientOutput,
    primal,
    gradient,
  )
where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Vector.Storable qualified as VS
import HordeAd
import HordeAd.Core.AstEnv
import HordeAd.Core.AstInterpret

data Input = Input
  { _inputA :: VS.Vector Double,
    _inputEll :: Int
  }

type PrimalOutput = Double

type GradientOutput = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "A" <*> o .: "ell"

chunk :: ADReady target
      => Int -> VS.Vector Double -> target (TKR 2 Double)
chunk n xs = rconcrete $ Nested.rfromVector [VS.length xs `div` n, n] xs

picks :: ADReady target
      => target (TKR 1 Double) -> target (TKR 2 Double)
picks l = rgather @2 [rwidth l, rwidth l - 1] l
                     (\ [n, n1] -> [ifH (n <=. n1) (n1 + 1) n1])

parts :: ADReady target
      => target (TKR 2 Double) -> target (TKR 3 Double)
parts m = rtr $ rbuild1 (rwidth m) (\i -> picks (m ! [i]))

minors :: ADReady target
       => target (TKR 2 Double) -> target (TKR 1 Double)
minors m =
  let parts_m = parts m
  in rbuild1 (rwidth parts_m) (\i -> rfromK $ det (parts_m ! [i]))

det :: ADReady target
    => target (TKR 2 Double) -> target (TKScalar Double)
det a | rwidth a == 1 = a `rindex0` [0, 0]
det a' = tlet a' $ \a ->
  let minors_a = minors $ rslice 1 (rwidth a - 1) a
      head_a = a ! [0]
      cycle1 = ringestData [rwidth minors_a] $ take (rwidth minors_a)
               $ cycle [1, -1]
  in rsum0 $ cycle1 * head_a * minors_a

primal :: Input -> PrimalOutput
primal (Input a ell) =
  let ast = simplifyInlineContract $ det (chunk ell a)
  in -- traceShow ("primal", printAstPrettyButNested ast) $
     unConcrete $ interpretAstFull emptyEnv ast

gradient :: Input -> GradientOutput
gradient (Input a ell) =
  Nested.rtoVector . unConcrete $ grad det (chunk ell a)
