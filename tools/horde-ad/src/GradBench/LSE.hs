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
logsumexpVS :: (NumScalar a, Differentiable a)
            => VS.Vector a -> a
logsumexpVS x = unConcrete
              . (logsumexp @1)
              . rconcrete . Nested.rfromVector [VS.length x] $ x

primal :: Input -> PrimalOutput
primal (Input x) = logsumexpVS x

gradient :: Input -> GradientOutput
gradient (Input x) = Nested.rtoVector . unConcrete
                     . grad (logsumexp @1)
                     . rconcrete . Nested.rfromVector [VS.length x] $ x
  -- cgrad and the symbolic grad are here equally fast
