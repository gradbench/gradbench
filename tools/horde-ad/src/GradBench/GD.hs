-- | This is not an eval, but rather a (simple) implementation of
-- Gradient Descent. It is very naive (e.g. using lists), but this is
-- sufficient for the 'particle' and 'saddle' evals.
module GradBench.GD
  ( multivariateArgmin,
    multivariateArgmax,
    multivariateMax,
    cgrad2_fwdR,
    cgrad_fwdK2,
  )
where

import GHC.TypeLits (KnownNat)
import HordeAd
import HordeAd.Core.Adaptor

square :: (NumScalar a, ADReady target)
       => target (TKR 1 a) -> target (TKR 1 a)
square x' = tlet x' $ \x -> x * x
-- slower even symbolically: square x = x ** rrepl (rshape x) 2

magnitude_squared :: (NumScalar a, ADReady target)
                  => target (TKR 1 a) -> target (TKR 0 a)
magnitude_squared = rsum0 . square

magnitude :: (NumScalar a, Differentiable a, ADReady target)
          => target (TKR 1 a) -> target (TKR 0 a)
magnitude = sqrt . magnitude_squared

scale :: (NumScalar a, ADReady target)
      => a -> target (TKR 1 a) -> target (TKR 1 a)
scale x v = rrepl (rshape v) x * v

distance_squared :: (NumScalar a, ADReady target)
                 => target (TKR 1 a) -> target (TKR 1 a) -> target (TKR 0 a)
distance_squared u v = magnitude_squared (u - v)

distance :: (NumScalar a, Differentiable a, ADReady target)
         => target (TKR 1 a) -> target (TKR 1 a) -> target (TKR 0 a)
distance u v = sqrt $ distance_squared u v

-- | The solver must be invoked with a function returning a pair: the cost
-- and the gradient.
multivariateArgmin
  :: (NumScalar a, Differentiable a, ADReady target, Ord (target (TKR 0 a)))
  => (target (TKR 1 a) -> (target (TKR 0 a), target (TKR 1 a)))
  -> target (TKR 1 a) -> target (TKR 1 a)
{-# INLINE multivariateArgmin #-}
multivariateArgmin fg x0 = loop (x0, fx0, gx0, 1e-5, 0 :: Int)
  where
    (fx0, gx0) = fg x0
    loop (x, fx, gx, eta, i)
      | magnitude gx <= rscalar 1e-5 = x
      | i == 10 = loop (x, fx, gx, 2 * eta, 0)
      | distance x x_prime <= rscalar 1e-5 = x
      | fx_prime < fx = loop (x_prime, fx_prime, gx_prime, eta, i + 1)
      | otherwise = loop (x, fx, gx, eta / 2, 0)
      where
        x_prime = x - (eta `scale` gx)
        (fx_prime, gx_prime) = fg x_prime

multivariateArgmax
  :: (NumScalar a, Differentiable a, ADReady target, Ord (target (TKR 0 a)))
  => (target (TKR 1 a) -> (target (TKR 0 a), target (TKR 1 a)))
  -> target (TKR 1 a) -> target (TKR 1 a)
{-# INLINE multivariateArgmax #-}
multivariateArgmax fg = multivariateArgmin (\arg -> let (c, g) = fg arg
                                                    in (-c, -g))

multivariateMax
  :: (NumScalar a, Differentiable a, ADReady target, Ord (target (TKR 0 a)))
  => (target (TKR 1 a) -> (target (TKR 0 a), target (TKR 1 a)))
  -> target (TKR 1 a) -> target (TKR 0 a)
{-# INLINE multivariateMax #-}
multivariateMax fg x = fst $ fg $ multivariateArgmax fg x

cgrad2_fwdR
  :: forall src r tgt target n.
     ( src ~ ADVal target (TKR n r)
     , NumScalar r, ADTensorScalar r ~ r, KnownNat n
     , tgt ~ ADVal target (TKScalar r)
     , ADReadyNoLet target, ShareTensor target
     , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
  => (src -> tgt)  -- ^ the objective function
  -> DValue src
  -> ( target (TKScalar r)
     , DValue src )  -- morally DValue (ADTensorKind src)
{-# INLINE cgrad2_fwdR #-}
cgrad2_fwdR f x =
  let sh = rshape $ fromDValue @src x
      g :: IxROf target n -> target (TKScalar r)
      g i = cjvp f x (roneHot sh (rscalar 1) i)
  in (kprimalPart $ f (fromDValue x), rbuild sh (rfromK . g))

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
