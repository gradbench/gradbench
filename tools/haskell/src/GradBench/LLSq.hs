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
import Data.Vector qualified as V
import Numeric.AD.Double qualified as D

data Input = Input
  { _inputX :: V.Vector Double,
    _inputN :: Int
  }

type PrimalOutput = Double

type GradientOutput = V.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (V.fromArray <$> o .: "x") <*> o .: "n"

t :: (Floating a) => Int -> Int -> a
t i n = (-1) + fromIntegral i * 2 / (fromIntegral n - 1)

primalPoly :: (Floating a) => V.Vector a -> Int -> a
primalPoly x n = 0.5 * sum (map f [0 .. n - 1])
  where
    f i =
      let ti = t i n
          g acc j xj = acc + negate (xj * ti ** fromIntegral j)
       in (signum ti + V.ifoldl' g 0 x) ** 2

primal :: Input -> PrimalOutput
primal (Input x n) = primalPoly x n

gradient :: Input -> GradientOutput
gradient (Input x n) = D.grad (`primalPoly` n) $ fmap D.auto x

{-# SPECIALIZE primalPoly :: V.Vector Double -> Int -> Double #-}
