{-# LANGUAGE DuplicateRecordFields, OverloadedRecordDot #-}
module GradBench.Particle (Input, Output, rr, fr, ff, rf) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import GradBench.GD (multivariateArgmin)
import Numeric.AD.Mode.Forward qualified as F
import Numeric.AD.Mode.Reverse qualified as R

data Input = Input Double

type Output = Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "w"

data Point a = Point {x :: a, y :: a}
  deriving (Functor, Traversable, Foldable)

pplus :: (Num a) => Point a -> Point a -> Point a
pplus u v = Point (u.x + v.x) (u.y + v.y)

ktimesp :: (Num a) => a -> Point a -> Point a
ktimesp k u = Point (k * u.x) (k * u.y)

sqr :: (Floating a) => a -> a
sqr x = x * x

dist :: (Floating a) => Point a -> Point a -> a
dist u v = sqrt (sqr (u.x - v.x) + sqr (u.y - v.y))

accel :: (Floating a) => [Point a] -> Point a -> a
accel charges x = sum $ map (\p -> 1 / dist p x) charges

{-# INLINE naiveEuler #-}
naiveEuler ::
  (Floating a, Ord a) =>
  ([Point a] -> Point a -> Point a) ->
  a ->
  a
naiveEuler accel' w =
  let x_initial = Point 0 8
      xdot_initial = Point 0.75 0
      (x, xdot) = loop x_initial xdot_initial
      delta_t_f = -x.y / xdot.y
      x_t_f = x `pplus` (delta_t_f `ktimesp` xdot)
   in sqr x_t_f.x
  where
    charges = [Point 10 (10 - w), Point 10 0]
    delta_t = 1e-1
    loop x xdot =
      let xddot = (-1) `ktimesp` accel' charges x
          x_new = x `pplus` (delta_t `ktimesp` xdot)
       in if x_new.y > 0
            then loop x_new $ xdot `pplus` (delta_t `ktimesp` xddot)
            else (x, xdot)

rr, rf, fr, ff :: Input -> Output
rr (Input w0) = head $ multivariateArgmin (f, g) [w0]
  where
    accel' charges = R.grad (accel $ map (fmap R.auto) charges)
    f w = naiveEuler accel' (head w)
    g = R.grad f
fr (Input w0) = head $ multivariateArgmin (f, g) [w0]
  where
    accel' charges = R.grad (accel $ map (fmap R.auto) charges)
    f w = naiveEuler accel' (head w)
    g = pure . F.diff (naiveEuler accel') . head
ff (Input w0) = head $ multivariateArgmin (f, g) [w0]
  where
    accel' charges p =
      Point
        (F.du (accel $ map (fmap F.auto) charges) (Point (p.x, 1) (p.y, 0)))
        (F.du (accel $ map (fmap F.auto) charges) (Point (p.x, 0) (p.y, 1)))
    f w = naiveEuler accel' (head w)
    g = pure . F.diff (naiveEuler accel') . head
rf (Input w0) = head $ multivariateArgmin (f, g) [w0]
  where
    accel' charges p =
      Point
        (F.du (accel $ map (fmap F.auto) charges) (Point (p.x, 1) (p.y, 0)))
        (F.du (accel $ map (fmap F.auto) charges) (Point (p.x, 0) (p.y, 1)))
    f w = naiveEuler accel' (head w)
    g = pure . R.diff (naiveEuler accel') . head
