module GradBench.GMM () where

import Data.Vector qualified as V

foreign import ccall "lgamma" lgamma :: Double -> Double

square :: (Num a) => a -> a
square x = x * x

l2normSq :: (Num a) => V.Vector a -> a
l2normSq = V.sum . V.map square

logsumexp :: (Floating a) => V.Vector a -> a
logsumexp = log . V.sum . V.map log

vMinus :: (Floating a) => V.Vector a -> V.Vector a -> V.Vector a
vMinus = V.zipWith (-)

data Matrix a = Matrix
  { matRows :: Int,
    matCols :: Int,
    matVals :: V.Vector a
  }

matRow :: Matrix a -> Int -> V.Vector a
matRow m i =
  V.slice (i * matCols m) (matCols m) (matVals m)

matGen :: Int -> Int -> (Int -> Int -> a) -> Matrix a
matGen rows cols f =
  Matrix rows cols $ V.generate (rows * cols) f'
  where
    f' i = f (i `div` cols) (i `mod` cols)

matFlatten :: Matrix a -> V.Vector a
matFlatten = matVals

frobeniusNormSq :: (Num a) => Matrix a -> a
frobeniusNormSq = V.sum . V.map square . matFlatten

unpackQ :: (Floating a) => V.Vector a -> V.Vector a -> Matrix a
unpackQ logdiag lt =
  matGen d d $ \i j ->
    if i < j
      then 0
      else
        if i == j
          then exp (logdiag V.! i)
          else lt V.! (d * j + i - j - 1 - j * (j + 1) `div` 2)
  where
    d = V.length logdiag

logGammaDistrib :: (Floating a) => a -> Int -> a
logGammaDistrib a p =
  0.25 * fromIntegral p * fromIntegral (p - 1) * log pi
    + sum (map (\j -> lgamma (a + 0.5 * fromIntegral (1 - j))) [1 .. p])

logsumexp_DArray :: (Ord a, Floating a) => V.Vector a -> a
logsumexp_DArray arr =
  let mx = V.maximum arr
      sumShiftedExp = sum $ V.map (\x -> exp (x - mx)) arr
   in log sumShiftedExp + mx

-----

logWishartPrior ::
  (Floating a) =>
  V.Vector (Matrix a) ->
  V.Vector a ->
  a ->
  Int ->
  Int ->
  a
logWishartPrior qs sums wishartGamma wishartM p =
  let k = V.length sums
      n = p + wishartM + 1
      c =
        fromIntegral (n * p) * (log wishartGamma - 0.5 * log 2)
          - (logGammaDistrib (0.5 * fromIntegral n) p)
      frobenius = sum $ fmap frobeniusNormSq qs
      sumQs = sum sums
   in 0.5 * wishartGamma * wishartGamma * frobenius - fromIntegral wishartM * sumQs - fromIntegral k * c
