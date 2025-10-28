{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.LLSq
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
  { _inputX :: VS.Vector Double,
    _inputN :: Int
  }

type PrimalOutput = Double

type GradientOutput = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (o .: "x") <*> o .: "n"

t :: (NumScalar a, Differentiable a, ADReady target)
  => IntOf target -> Int -> target (TKR 0 a)
t i n = negate (rscalar 1)
        + rfromIndex0 i * rscalar 2 / (rscalar (fromIntegral n) - rscalar 1)

primalPoly :: (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> Int -> target (TKScalar a)
primalPoly x n = kfromR $ rscalar 0.5 * rsum0 (rbuild1 n f)
 where
  f i = tlet (t i n) $ \ti ->
    let muls = rscan (*) (rscalar 1) $ rreplicate (rwidth x - 1) ti
    in (signum ti - rsum0 (muls * x)) ** rscalar 2

primal :: Input -> PrimalOutput
primal (Input x n) =
  let y = rconcrete . Nested.rfromVector [VS.length x] $ x
  in unConcrete $ primalPoly y n
    -- this is slow, because computed non-sybolically but with rbuild1
    -- TODO: if this matters for the score, force the symbolic pipeline

gradient :: Input -> GradientOutput
gradient (Input x n) =
  let y = rconcrete . Nested.rfromVector [VS.length x] $ x
  in Nested.rtoVector . unConcrete
     $ grad (`primalPoly` n) y
    -- non-symbolic cgrad would take much more memory and time here
    -- due to rbuild1 above
