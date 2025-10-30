{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
-- | This is an implementation that directly implements the recursive
-- specification and so is fundamentally so inefficient (something like O(n!)). -- Fortunately, the workloads are necessarily tiny so this doesn't OOM.
-- Due to the tiny workloads and, consequently, numerous but tiny tensors,
-- the gradient but especially the primal are much slower than
-- when implemented with lists for haskell-ad.
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
  in rbuild1 (rwidth parts_m) (\i -> det (parts_m ! [i]))

det :: ADReady target
    => target (TKR 2 Double) -> target (TKR 0 Double)
det a | rsize a == 1 = a ! [0, 0]  -- needed to avoid "rbuild1: shape ambiguity"
det a' = tlet a' $ \a ->
  let minors_a = minors $ rslice 1 (rwidth a - 1) a
      head_a = a ! [0]
      cycle1 = ringestData [rwidth minors_a] $ take (rwidth minors_a)
               $ cycle [1, -1]
  in rsum0 $ cycle1 * head_a * minors_a

primal :: Input -> PrimalOutput
primal (Input a ell) = unConcrete $ kfromR . det $ chunk ell a

gradient :: Input -> GradientOutput
gradient (Input a ell) =
  Nested.rtoVector . unConcrete $ grad (kfromR . det) (chunk ell a)
    -- non-symbolic cgrad would take much more memory and time here
    -- due to rbuild1 above
