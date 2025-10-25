{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.ODE
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
    _inputS :: Int
  }

type PrimalOutput = VS.Vector Double

type GradientOutput = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (o .: "x") <*> o .: "s"

scale :: (NumScalar a, ADReady target)
      => a -> target (TKR 1 a) -> target (TKR 1 a)
scale x v = rrepl (rshape v) x * v

rungeKutta
  :: (NumScalar a, Differentiable a, ADReady target)
  => (target (TKR 1 a) -> target (TKR 1 a))
  -> target (TKR 1 a)
  -> a
  -> Int
  -> target (TKR 1 a)
rungeKutta f yi tf s =
  loop s yi
 where
  loop 0 !yf = yf
  loop i yf' =
    tlet yf' $ \yf ->
    tlet (f yf) $ \k1 ->
    tlet (f $ yf + scale (h / 2) k1) $ \k2 ->
    tlet (f $ yf + scale (h / 2) k2) $ \k3 ->
      let k4 = f $ yf + scale h k3
      in loop (i - 1)
         $ yf + scale (h / 6) (k1
                               + scale 2 k2
                               + scale 2 k3
                               + k4)
  h = tf / fromIntegral s

-- Vectors x and y are assumed to have the same size.
-- Argument x is assumed to be duplicable (ie. shared outside this function).
odeFun :: (NumScalar a, ADReady target)
       => target (TKR 1 a) -> target (TKR 1 a) -> target (TKR 1 a)
odeFun x y =
  rappend (rslice 0 1 x) (rslice 1 (rwidth x - 1) x * rslice 0 (rwidth x - 1) y)

primalPoly :: (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> Int -> target (TKR 1 a)
primalPoly x' s = tlet x' $ \x ->  -- shared to avoid lots of sharing in odeFun
  rungeKutta (odeFun x) (rrepl (rshape x) 0) tf s
 where
  tf = 2

primal :: Input -> PrimalOutput
primal (Input x s) =
  Nested.rtoVector $ unConcrete
  $ primalPoly (rconcrete $ Nested.rfromVector [VS.length x] x) s

gradient :: Input -> GradientOutput
gradient (Input x s) =
  let f a = let res = primalPoly a s
            in kfromR $ res ! [fromIntegral $ rwidth res - 1]
  in Nested.rtoVector $ unConcrete
     $ grad f (rconcrete $ Nested.rfromVector [VS.length x] x)
