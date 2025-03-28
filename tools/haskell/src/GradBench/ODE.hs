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
import Data.Vector qualified as V
import Numeric.AD.Double qualified as D

data Input = Input
  { _inputX :: V.Vector Double,
    _inputS :: Int
  }

type PrimalOutput = V.Vector Double

type GradientOutput = V.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (V.fromArray <$> o .: "x") <*> o .: "s"

vecadd :: (Num a) => V.Vector a -> V.Vector a -> V.Vector a
vecadd = V.zipWith (+)

scale :: (Num a) => a -> V.Vector a -> V.Vector a
scale x v = V.map (x *) v

rungeKutta ::
  (Floating a) =>
  (V.Vector a -> V.Vector a) ->
  V.Vector a ->
  a ->
  Int ->
  V.Vector a
rungeKutta f yi tf s =
  loop s yi
  where
    loop 0 yf = yf
    loop i yf =
      let k1 = f yf
          k2 = f $ yf `vecadd` scale (h / 2) k1
          k3 = f $ yf `vecadd` scale (h / 2) k2
          k4 = f $ yf `vecadd` scale h k3
       in loop (i - 1) $
            yf
              `vecadd` scale
                (h / 6)
                ( k1
                    `vecadd` scale 2 k2
                    `vecadd` scale 2 k3
                    `vecadd` k4
                )
    h = tf / fromIntegral s

odeFun :: (Num a) => V.Vector a -> V.Vector a -> V.Vector a
odeFun x y =
  V.cons (V.head x) (V.zipWith (*) (V.tail x) (V.init y))

primalPoly :: (Floating a) => V.Vector a -> Int -> V.Vector a
primalPoly x s = rungeKutta (odeFun x) (V.map (const 0) x) tf s
  where
    tf = 2

primal :: Input -> PrimalOutput
primal (Input x s) = primalPoly x s

gradient :: Input -> PrimalOutput
gradient (Input x s) = D.grad (V.last . flip primalPoly s) x
