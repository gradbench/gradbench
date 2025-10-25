{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.LSE
  ( primal,
    gradient,
  )
where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Vector qualified as V
import HordeAd

newtype Input = Input (V.Vector Double)

type PrimalOutput = Double

-- TODO: use unboxed vectors instead, if the harness permits it
type GradientOutput = V.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (V.fromArray <$> o .: "x")

-- Fails for empty argument.
logsumexp :: (NumScalar a, Differentiable a)
          => V.Vector a -> a
logsumexp x = unConcrete
              . logsumexpTarget
              . ringestData [V.length x] . V.toList $ x

logsumexpTarget :: (NumScalar a, Differentiable a, ADReady target)
                => target (TKR 1 a) -> target (TKScalar a)
logsumexpTarget x =
  kfromR $ (+ a) $ log $ rsum $ exp . subtract (rreplicate k a) $ x
 where
  a = rmaximum x  -- fails for empty x
  k = rsize x

primal :: Input -> PrimalOutput
primal (Input x) = logsumexp x

gradient :: Input -> GradientOutput
gradient (Input x) = V.fromList . Nested.rtoList . unConcrete
                     . grad logsumexpTarget
                     . ringestData [V.length x] . V.toList $ x
