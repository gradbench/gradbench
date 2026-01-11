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
{-# INLINE rungeKutta #-}
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
         $ yf + scale (h / 6) (k1 + k4)
              + scale (h / 3) (k2 + k3)
  h = tf / fromIntegral s

odeFun :: (NumScalar a, ADReady target)
       => target (TKR 1 a) -> target (TKR 1 a) -> target (TKR 1 a)
       -> target (TKR 1 a)
odeFun x0 xn y = rappend x0 (xn * rslice 0 (rwidth xn) y)

primalPoly :: (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> Int -> target (TKR 1 a)
primalPoly x' s =
  tlet x' $ \x ->
  tlet (rslice 0 1 x) $ \x0 ->  -- shared across many calls to odeFun
  tlet (rslice 1 (rwidth x - 1) x) $ \xn ->
    rungeKutta (odeFun x0 xn) (rrepl (rshape x) 0) tf s
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
     $ cgrad f (rconcrete $ Nested.rfromVector [VS.length x] x)
    -- cgrad is here many times faster than the symbolic grad
    -- for tests with s (loop count) much higher than n (data size)
