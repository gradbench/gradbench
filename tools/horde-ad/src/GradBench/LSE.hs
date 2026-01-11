{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.LSE
  ( primal,
    gradient,
  )
where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Vector.Storable qualified as VS
import HordeAd

newtype Input = Input (VS.Vector Double)

type PrimalOutput = Double

type GradientOutput = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (o .: "x")

-- Fails for empty argument.
logsumexp :: (NumScalar a, Differentiable a)
          => VS.Vector a -> a
logsumexp x = unConcrete
              . logsumexpTarget
              . rconcrete . Nested.rfromVector [VS.length x] $ x

logsumexpTarget :: (NumScalar a, Differentiable a, ADReady target)
                => target (TKR 1 a) -> target (TKScalar a)
logsumexpTarget x' =
  tlet x' $ \x ->
  tlet (rmaximum x) $ \a ->  -- fails for empty x
    kfromR . (+ a) . log . rsum . exp . subtract (rreplicate (rwidth x) a) $ x

primal :: Input -> PrimalOutput
primal (Input x) = logsumexp x

gradient :: Input -> GradientOutput
gradient (Input x) = Nested.rtoVector . unConcrete
                     . grad logsumexpTarget
                     . rconcrete . Nested.rfromVector [VS.length x] $ x
  -- cgrad and the symbolic grad are here equally fast
