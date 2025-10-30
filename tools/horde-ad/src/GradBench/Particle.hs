{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.Particle (Input, Output, rr, ff, fr, rf) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import GradBench.GD (cgrad2_fwdR, cgrad_fwdK2, multivariateArgmin)
import HordeAd

data Input = Input Double

type Output = Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "w"

type Point a = (a, a)

pplus :: (Num a) => Point a -> Point a -> Point a
pplus u v = (fst u + fst v, snd u + snd v)

ktimesp :: (Num a) => a -> Point a -> Point a
ktimesp k u = (k * fst u, k * snd u)

sqr :: (Floating a) => a -> a
sqr x = x * x

dist :: (Floating a) => Point a -> Point a -> a
dist u v = sqrt (sqr (fst u - fst v) + sqr (snd u - snd v))

accel :: (Floating a) => [Point a] -> Point a -> a
accel charges x = sum $ map (\p -> recip (dist p x)) charges

naiveEuler
  :: (ADReadyNoLet target, Ord (target (TKScalar Double)))
  => ([Point (target (TKScalar Double))] -> Point (target (TKScalar Double))
      -> Point (target (TKScalar Double)))
  -> target (TKScalar Double)
  -> target (TKScalar Double)
{-# INLINE naiveEuler #-}
naiveEuler accel' w =
  let x_initial = (0, 8)
      xdot_initial = (0.75, 0)
      (x, xdot) = loop x_initial xdot_initial
      delta_t_f = - (snd x) / snd xdot
      x_t_f = x `pplus` (delta_t_f `ktimesp` xdot)
   in sqr (fst x_t_f)
 where
  charges = [(10, (10 - w)), (10, 0)]
  delta_t = 1e-1
  loop x xdot =
    let xddot = (-1) `ktimesp` accel' charges x
        x_new = x `pplus` (delta_t `ktimesp` xdot)
    in if snd x_new > 0
       then loop x_new $ xdot `pplus` (delta_t `ktimesp` xddot)
       else (x, xdot)

-- TODO: this is very slow; see the comment in Saddle.hs
rr, ff, fr, rf :: Input -> Output
rr (Input w0) = unConcrete $ kfromR
                $ multivariateArgmin g (rrepl [1] w0) ! [0]
  where
    accel' :: forall target.
              ( ADReadyNoLet target, ShareTensor target
              , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
           => [Point (target (TKScalar Double))]
           -> Point (target (TKScalar Double))
           -> Point (target (TKScalar Double))
    accel' charges = cgrad @_ @_ @_ @target
                           (accel $ map (\(x, y) ->
                              (kfromPrimal x, kfromPrimal y)) charges)
    f :: ( ADReadyNoLet target, ShareTensor target
         , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
         , Ord (target (TKScalar Double)) )
      => target (TKR 1 Double) -> target (TKScalar Double)
    f w = naiveEuler accel' (kfromR $ w ! [0])
    g :: ( ADReadyNoLet target, ShareTensor target
         , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
         , Ord (target (TKScalar Double)) )
      => target (TKR 1 Double)
      -> (target (TKR 0 Double), target (TKR 1 Double))
    g a = let (res0, res1) = cgrad2 f a
          in (rfromK res0, res1)
ff (Input w0) = unConcrete $ kfromR
                $ multivariateArgmin g (rrepl [1] w0) ! [0]
  where
    accel' :: forall target.
              ( ADReadyNoLet target, ShareTensor target
              , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
           => [Point (target (TKScalar Double))]
           -> Point (target (TKScalar Double))
           -> Point (target (TKScalar Double))
    accel' charges = cgrad_fwdK2 @_ @_ @_ @target
                                 (accel $ map (\(x, y) ->
                                    (kfromPrimal x, kfromPrimal y)) charges)
    f :: ( ADReadyNoLet target, ShareTensor target
         , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
         , Ord (target (TKScalar Double)) )
      => target (TKR 1 Double) -> target (TKScalar Double)
    f w = naiveEuler accel' (kfromR $ w ! [0])
    g :: ( ADReadyNoLet target, ShareTensor target
         , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
         , Ord (target (TKScalar Double)) )
      => target (TKR 1 Double)
      -> (target (TKR 0 Double), target (TKR 1 Double))
    g a = let (res0, res1) = cgrad2_fwdR f a
          in (rfromK res0, res1)
fr (Input w0) = unConcrete $ kfromR
                $ multivariateArgmin g (rrepl [1] w0) ! [0]
  where
    accel' :: forall target.
              ( ADReadyNoLet target, ShareTensor target
              , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
           => [Point (target (TKScalar Double))]
           -> Point (target (TKScalar Double))
           -> Point (target (TKScalar Double))
    accel' charges = cgrad @_ @_ @_ @target
                           (accel $ map (\(x, y) ->
                              (kfromPrimal x, kfromPrimal y)) charges)
    f :: ( ADReadyNoLet target, ShareTensor target
         , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
         , Ord (target (TKScalar Double)) )
      => target (TKR 1 Double) -> target (TKScalar Double)
    f w = naiveEuler accel' (kfromR $ w ! [0])
    g :: ( ADReadyNoLet target, ShareTensor target
         , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
         , Ord (target (TKScalar Double)) )
      => target (TKR 1 Double)
      -> (target (TKR 0 Double), target (TKR 1 Double))
    g a = let (res0, res1) = cgrad2_fwdR f a
          in (rfromK res0, res1)
rf (Input w0) = unConcrete $ kfromR
                $ multivariateArgmin g (rrepl [1] w0) ! [0]
  where
    accel' :: forall target.
              ( ADReadyNoLet target, ShareTensor target
              , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
           => [Point (target (TKScalar Double))]
           -> Point (target (TKScalar Double))
           -> Point (target (TKScalar Double))
    accel' charges = cgrad_fwdK2 @_ @_ @_ @target
                                 (accel $ map (\(x, y) ->
                                    (kfromPrimal x, kfromPrimal y)) charges)
    f :: ( ADReadyNoLet target, ShareTensor target
         , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
         , Ord (target (TKScalar Double)) )
      => target (TKR 1 Double) -> target (TKScalar Double)
    f w = naiveEuler accel' (kfromR $ w ! [0])
    g :: ( ADReadyNoLet target, ShareTensor target
         , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
         , Ord (target (TKScalar Double)) )
      => target (TKR 1 Double)
      -> (target (TKR 0 Double), target (TKR 1 Double))
    g a = let (res0, res1) = cgrad2 f a
          in (rfromK res0, res1)
