{-# LANGUAGE DuplicateRecordFields, OverloadedRecordDot #-}
module GradBench.Saddle (Input, Output, rr, ff, rf, fr) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import GradBench.GD (multivariateArgmax, multivariateArgmin, multivariateMax)
import Numeric.AD.Mode.Forward qualified as F
import Numeric.AD.Mode.Reverse qualified as R

data Input = Input (Double, Double)

type Output = [Double]

data Point a = Point {x :: a, y :: a}
  deriving (Functor, Traversable, Foldable)

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "start"

f :: (Floating a) => Point a -> Point a -> a
f p1 p2 = (p1.x ** 2 + p1.y ** 2) - (p2.x ** 2 + p2.y ** 2)

{-# INLINE saddleGen #-}
saddleGen ::
  (Ord a, Floating a) =>
  ([a] -> [a] -> a) ->
  ([a] -> [a]) ->
  ([a] -> [a] -> a) ->
  ([a] -> [a] -> [a]) ->
  [a] ->
  [a]
saddleGen r1cost r1cost' r2cost r2cost' start =
  let r1 = multivariateArgmin (r1cost start, r1cost') start
      r2 = multivariateArgmax (r2cost r1, r2cost' r1) start
   in r1 ++ r2

rr, rf, fr, ff :: Input -> Output
rr (Input (x, y)) = saddleGen r1cost r1cost' r2cost r2cost' start
  where
    start = [x, y]
    r1cost start' p1 = multivariateMax (r2cost p1, r2cost' p1) start'
    r1cost' = R.grad $ r1cost $ map R.auto start
    r2cost r1 r2 = f (Point (r1 !! 0) (r1 !! 1)) (Point (r2 !! 0) (r2 !! 1))
    r2cost' r1 = R.grad $ r2cost $ map R.auto r1
ff (Input (x, y)) = saddleGen r1cost r1cost' r2cost r2cost' start
  where
    start = [x, y]
    r1cost start' p1 = multivariateMax (r2cost p1, r2cost' p1) start'
    r1cost' = F.grad $ r1cost $ map F.auto start
    r2cost r1 r2 = f (Point (r1 !! 0) (r1 !! 1)) (Point (r2 !! 0) (r2 !! 1))
    r2cost' r1 = F.grad $ r2cost $ map F.auto r1
fr (Input (x, y)) = saddleGen r1cost r1cost' r2cost r2cost' start
  where
    start = [x, y]
    r1cost start' p1 = multivariateMax (r2cost p1, r2cost' p1) start'
    r1cost' = F.grad $ r1cost $ map F.auto start
    r2cost r1 r2 = f (Point (r1 !! 0) (r1 !! 1)) (Point (r2 !! 0) (r2 !! 1))
    r2cost' r1 = R.grad $ r2cost $ map R.auto r1
rf (Input (x, y)) = saddleGen r1cost r1cost' r2cost r2cost' start
  where
    start = [x, y]
    r1cost start' p1 = multivariateMax (r2cost p1, r2cost' p1) start'
    r1cost' = R.grad $ r1cost $ map R.auto start
    r2cost r1 r2 = f (Point (r1 !! 0) (r1 !! 1)) (Point (r2 !! 0) (r2 !! 1))
    r2cost' r1 = F.grad $ r2cost $ map F.auto r1
