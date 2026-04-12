{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.Particle (Input, Output, rr, ff, fr, rf) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import GradBench.GD
import HordeAd
import HordeAd.Core.Adaptor

newtype Input = Input Double

type Output = Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "w"

magnitude_squaredK :: (NumScalar a, ADReady target)
                   => target (TKScalar a) -> target (TKScalar a)
{-# INLINE magnitude_squaredK #-}
magnitude_squaredK t' = tlet t' $ \t -> t * t

scaleK :: (NumScalar a, ADReady target)
       => a -> target (TKScalar a) -> target (TKScalar a)
{-# INLINE scaleK #-}
scaleK x v = kconcrete x * v

cgrad_fwdK2
  :: forall src r tgt target.
     ( src ~ (ADVal target (TKScalar r), ADVal target (TKScalar r))
     , NumScalar r, ADTensorScalar r ~ r
     , tgt ~ ADVal target (TKScalar r)
     , ADReadyNoLet target, ShareTensor target
     , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
  => (src -> tgt)  -- ^ the objective function
  -> DValue src
  -> DValue src  -- morally DValue (ADTensorKind src)
{-# INLINE cgrad_fwdK2 #-}
cgrad_fwdK2 f x = (cjvp f x (1, 0), cjvp f x (0, 1))

cgrad2_fwdK
  :: forall src r tgt target.
     ( src ~ ADVal target (TKScalar r)
     , NumScalar r, ADTensorScalar r ~ r
     , tgt ~ ADVal target (TKScalar r)
     , ADReadyNoLet target, ShareTensor target )
  => (src -> tgt)  -- ^ the objective function
  -> DValue src
  -> ( target (TKScalar r)
     , DValue src )  -- morally DValue (ADTensorKind src)
{-# INLINE cgrad2_fwdK #-}
cgrad2_fwdK f x = cjvp2 f x 1

type Point a = (a, a)

-- No sharing, so not good for the symbolic pipeline, but adequate here.
pplus :: (Num a) => Point a -> Point a -> Point a
{-# INLINE pplus #-}
pplus u v = (fst u + fst v, snd u + snd v)

pminus :: (Num a) => Point a -> Point a -> Point a
{-# INLINE pminus #-}
pminus u v = (fst u - fst v, snd u - snd v)

ktimesp :: (Num a) => a -> Point a -> Point a
{-# INLINE ktimesp #-}
ktimesp k u = (k * fst u, k * snd u)

sqr :: (Floating a) => a -> a
{-# INLINE sqr #-}
sqr x = x * x

dist :: (Floating a) => Point a -> Point a -> a
{-# INLINE dist #-}
dist u v = sqrt (sqr (fst u - fst v) + sqr (snd u - snd v))

accel :: (Floating a) => [Point a] -> Point a -> a
{-# INLINE accel #-}
accel charges x =
  foldl' (\ !acc p@(!_, !_) -> acc + recip (dist p x)) 0 charges

naiveEuler
  :: forall a target. a ~ Double
  => (BaseTensor target, Ord (target (TKScalar a)))
  => ([Point (target (TKScalar a))] -> Point (target (TKScalar a))
      -> Point (target (TKScalar a)))
  -> target (TKScalar a)
  -> target (TKScalar a)
{-# INLINE naiveEuler #-}
naiveEuler accel' w =
  let x_initial = (0, 8)
      xdot_initial = (0.75, 0)
      (x, xdot) = loop x_initial xdot_initial
      delta_t_f = snd x / snd xdot
      x_t_f = x `pminus` (delta_t_f `ktimesp` xdot)
  in sqr (fst x_t_f)
 where
  charges = [(10, 10 - w), (10, 0)]
  delta_t = 1e-1
  loop x@(!_, !_) xdot@(!_, !_) =
    let xddot = accel' charges x
        x_new = x `pplus` (delta_t `ktimesp` xdot)
    in if snd x_new > 0
       then loop x_new (xdot `pminus` (delta_t `ktimesp` xddot))
       else (x, xdot)

particleGen
  :: forall a. a ~ Double
  => (forall target.
      ( ADReadyNoLet target, ShareTensor target
      , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
      => ((ADVal target (TKScalar a), ADVal target (TKScalar a))
          -> ADVal target (TKScalar a))
      -> (target (TKScalar a), target (TKScalar a))
      -> (target (TKScalar a), target (TKScalar a)))
  -> (forall target.
      (ADReadyNoLet target, ShareTensor target)
      => (ADVal target (TKScalar a)
          -> ADVal target (TKScalar a))
      -> target (TKScalar a)
      -> (target (TKScalar a), target (TKScalar a)))
  -> Input
  -> Output
{-# INLINE particleGen #-}
particleGen cgradA cgrad2B (Input w0) =
  unConcrete
  $ multivariateArgmin magnitude_squaredK scaleK g (kconcrete w0)
 where
  accel' :: ( ADReadyNoLet target, ShareTensor target
            , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
         => [Point (target (TKScalar a))] -> Point (target (TKScalar a))
         -> Point (target (TKScalar a))
  accel' charges = cgradA (accel $ map (\(x, y) ->
                             (kfromPrimal x, kfromPrimal y)) charges)
  f :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar a)) )
    => target (TKScalar a) -> target (TKScalar a)
  f = naiveEuler accel'
  g :: ( ADReadyNoLet target, ShareTensor target
       , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
       , Ord (target (TKScalar a)) )
    => target (TKScalar a) -> (target (TKScalar a), target (TKScalar a))
  g = cgrad2B f

-- TODO: this is very slow; see the comment in Saddle.hs
rr, ff, fr, rf :: Input -> Output
rr = particleGen cgrad       cgrad2
ff = particleGen cgrad_fwdK2 cgrad2_fwdK
fr = particleGen cgrad       cgrad2_fwdK
rf = particleGen cgrad_fwdK2 cgrad2
