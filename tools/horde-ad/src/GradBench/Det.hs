{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
-- | This implementation is based on det.fut, which is based on
--   https://mathworld.wolfram.com/DeterminantExpansionbyMinors.html
module GradBench.Det
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
import Data.Int (Int64)
import Data.Vector.Storable qualified as VS
import HordeAd

data Input = Input
  { _inputA :: VS.Vector Double,
    _inputEll :: Int
  }

type PrimalOutput = Double

type GradientOutput = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "A" <*> o .: "ell"

chunk :: ADReady target
      => Int -> VS.Vector Double -> target (TKR 2 Double)
chunk n xs = rconcrete $ Nested.rfromVector [VS.length xs `div` n, n] xs

fact :: Int -> Int
fact n = factAcc 1 n
 where factAcc acc i | i <= 1 = acc
       factAcc acc i = factAcc (i * acc) (i - 1)

-- Given the lexicographic index of a permutation, compute that
-- permutation.
idx_to_perm :: Int -> Int -> [Int]
idx_to_perm len idx0 =
  let perm0 = replicate len (-1)
      elements0 = [0 .. len - 1]
      fi0 = fact len
      allFreeSpots :: [Int] -> [Int]
      allFreeSpots elements1 =
        let f (-1) acc = acc
            f el acc = el : acc
        in foldr f [] elements1
      loop :: [Int] -> [Int] -> Int -> Int -> Int -> [Int]
      loop perm _ _ _ (-1) = perm
      loop perm elements idx fi i =
        let fi2 = fi `div` (i + 1)
            el = allFreeSpots elements !! (idx `div` fi2)
        in loop (take (len - i - 1) perm ++ [el] ++ drop (len - i) perm)
                (take el elements ++ [-1] ++ drop (el + 1) elements)
                (idx `mod` fi2)
                fi2
                (i - 1)
  in loop perm0 elements0 idx0 fi0 (len - 1)

-- Compute the inversion number from a lexicographic index of a
-- permutation.
inversion_number_from_idx :: Int -> Int -> Int
inversion_number_from_idx n idx0 =
  let loop s _ _ i | i == 0 = s
      loop s idx fi i =
        let fi2 = fi `div` (i + 1)
        in loop (s + idx `div` fi2) (idx `mod` fi2) fi2 (i - 1)
  in loop 0 idx0 (fact n) (n - 1)

productR :: ADReady target
         => target (TKR 1 Double) -> target (TKScalar Double)
productR = kfromR . rfold (*) (rscalar 1)

det :: forall target. ADReady target
    => target (TKR 2 Double) -> target (TKScalar Double)
det a =
  let ell = rwidth a
      f :: Int -> target (TKScalar Double)
      f i =
        let p :: PlainOf target (TKR 1 Int64)
            p = ringestData [ell] $ map fromIntegral $ idx_to_perm ell i
        in (-1) ** kconcrete (fromIntegral $ inversion_number_from_idx ell i)
           * productR (rbuild1 ell $ \i2 -> a ! [i2, kfromR $ p ! [i2]])
      g :: Int -> target (TKScalar Double) -> target (TKScalar Double)
      g i acc = acc + f i
  in foldr @[] g 0 [0, 1 .. fact ell - 1]

primal :: Input -> PrimalOutput
primal (Input a ell) = unConcrete $ det $ chunk ell a

gradient :: Input -> GradientOutput
gradient (Input a ell) =
  Nested.rtoVector . unConcrete $ cgrad det (chunk ell a)
