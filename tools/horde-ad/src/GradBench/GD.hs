-- | This is not an eval, but rather a (simple) implementation of
-- Gradient Descent. It is very naive (e.g. using lists), but this is
-- sufficient for the 'particle' and 'saddle' evals.
module GradBench.GD
  ( multivariateArgmin,
    multivariateArgmax,
    multivariateMax,
  )
where

import HordeAd

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

-- | The solver must be invoked with a pair of functions: the cost
-- function and its gradient.
multivariateArgmin
  :: (NumScalar a, Differentiable a)
  => ( Concrete (TKR 1 a) -> Concrete (TKR 0 a)
     , Concrete (TKR 1 a) -> Concrete (TKR 1 a) )
  -> Concrete (TKR 1 a) -> Concrete (TKR 1 a)
multivariateArgmin (f, g) x0 = loop (x0, f x0, g x0, 1e-5, 0 :: Int)
  where
    loop (x, fx, gx, eta, i)
      | magnitude gx <= 1e-5 = x
      | i == 10 = loop (x, fx, gx, 2 * eta, 0)
      | distance x x_prime <= 1e-5 = x
      | fx_prime < fx = loop (x_prime, fx_prime, g x_prime, eta, i + 1)
      | otherwise = loop (x, fx, gx, eta / 2, 0)
      where
        x_prime = x - (eta `scale` gx)
        fx_prime = f x_prime

multivariateArgmax
  :: (NumScalar a, Differentiable a)
  => ( Concrete (TKR 1 a) -> Concrete (TKR 0 a)
     , Concrete (TKR 1 a) -> Concrete (TKR 1 a) )
  -> Concrete (TKR 1 a) -> Concrete (TKR 1 a)
multivariateArgmax (f, g) x = multivariateArgmin (negate . f, negate . g) x

multivariateMax
  :: (NumScalar a, Differentiable a)
  => ( Concrete (TKR 1 a) -> Concrete (TKR 0 a)
     , Concrete (TKR 1 a) -> Concrete (TKR 1 a) )
  -> Concrete (TKR 1 a) -> Concrete (TKR 0 a)
multivariateMax (f, g) x = f $ multivariateArgmax (f, g) x

{-# INLINE multivariateArgmin #-}

{-# INLINE multivariateArgmax #-}

{-# INLINE multivariateMax #-}
