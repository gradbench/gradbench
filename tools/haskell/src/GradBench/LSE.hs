module GradBench.LSE
  ( primal,
    gradient,
  )
where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Vector qualified as V
import Numeric.AD.Double qualified as D

newtype Input = Input (V.Vector Double)

type PrimalOutput = Double

type GradientOutput = V.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (V.fromArray <$> o .: "x")

logsumexp :: (Ord a, Floating a) => V.Vector a -> a
logsumexp x =
  (+ a) $ log $ V.sum $ V.map (exp . subtract a) x
  where
    a = V.maximum x

primal :: Input -> PrimalOutput
primal (Input x) = logsumexp x

gradient :: Input -> GradientOutput
gradient (Input x) = D.grad logsumexp $ fmap D.auto x
