-- | This is not an eval, but rather a (simple) implementation of
-- Gradient Descent. It is very naive (e.g. using lists), but this is
-- sufficient for the 'particle' and 'saddle' evals.
module GradBench.GD
  ( multivariateArgmin,
    multivariateArgmax,
    multivariateMax,
  )
where

magnitude_squared :: (Floating a) => [a] -> a
magnitude_squared = sum . map (\x -> x * x)

magnitude :: (Floating a) => [a] -> a
magnitude = sqrt . magnitude_squared

vminus :: (Num a) => [a] -> [a] -> [a]
vminus = zipWith (-)

ktimesv :: (Num a) => a -> [a] -> [a]
ktimesv k = map (k *)

distance_squared :: (Floating a) => [a] -> [a] -> a
distance_squared u v = magnitude_squared (u `vminus` v)

distance :: (Floating a) => [a] -> [a] -> a
distance u v = sqrt $ distance_squared u v

-- | The solver must be invoked with a pair of functions: the cost
-- function and its gradient.
multivariateArgmin ::
  (Ord a, Floating a) =>
  ([a] -> a, [a] -> [a]) ->
  [a] ->
  [a]
multivariateArgmin (f, g) x0 = loop (x0, f x0, g x0, 1e-5, 0 :: Int)
  where
    loop (x, fx, gx, eta, i)
      | magnitude gx <= 1e-5 = x
      | i == 10 = loop (x, fx, gx, 2 * eta, 0)
      | distance x x_prime <= 1e-5 = x
      | fx_prime < fx = loop (x_prime, fx_prime, g x_prime, eta, i + 1)
      | otherwise = loop (x, fx, gx, eta / 2, 0)
      where
        x_prime = x `vminus` (eta `ktimesv` gx)
        fx_prime = f x_prime

multivariateArgmax ::
  (Ord a, Floating a) =>
  ([a] -> a, [a] -> [a]) ->
  [a] ->
  [a]
multivariateArgmax (f, g) x = multivariateArgmin (negate . f, map negate . g) x

multivariateMax ::
  (Ord a, Floating a) =>
  ([a] -> a, [a] -> [a]) ->
  [a] ->
  a
multivariateMax (f, g) x = f $ multivariateArgmax (f, g) x

{-# INLINE multivariateArgmin #-}

{-# INLINE multivariateArgmax #-}

{-# INLINE multivariateMax #-}
