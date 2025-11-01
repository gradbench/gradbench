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

updateR :: ADReady target
        => Int
        -> target (TKR 0 Int64) -> target (TKR 1 Int64)
        -> target (TKR 1 Int64)
updateR i a v = rslice 0 i v
                `rappend` rreplicate 1 a
                `rappend` rslice (i + 1) (rwidth v - i - 1) v

updateI :: ADReady target
        => PlainOf target (TKScalar Int64)
        -> target (TKR 0 Int64) -> target (TKR 1 Int64)
        -> target (TKR 1 Int64)
updateI idx a v =
  rgather1 (rwidth v) (v `rappend` rreplicate 1 a) $ \i ->
    [ifH (i ==. idx) (fromIntegral $ rwidth v) i]

-- Given the lexicographic index of a permutation, compute that
-- permutation.
idx_to_perm :: forall target. ADReady target
            => Int -> PlainOf target (TKScalar Int64) -> target (TKR 1 Int64)
idx_to_perm len idx0 =
  let dummyEl = rscalar (-1)
      occupiedSpotMark :: forall f. ADReady f => f (TKR 0 Int64)
      occupiedSpotMark = rscalar (-1)
      perm0 = rreplicate len dummyEl
      elements0 = riota len
      fi0 = fact len
      allFreeSpots :: target (TKR 1 Int64) -> target (TKR 1 Int64)
      allFreeSpots elements1 =
        let f :: forall f. ADReady f
              => f (TKProduct (TKR 1 Int64) (TKScalar Int64)) -> f (TKR 0 Int64)
              -> f (TKProduct (TKR 1 Int64) (TKScalar Int64))
            f acc el =
              ifH (el ==. occupiedSpotMark)
                  acc
                  (tpair
                     (updateI (tplainPart $ tproject2 acc) el (tproject1 acc))
                     (tproject2 acc + 1))
        in withSNat len $ \k ->
             tproject1
             $ tfold k knownSTK knownSTK f (tpair (rreplicate len dummyEl) 0)
                     elements1
      loop :: target (TKR 1 Int64) -> target (TKR 1 Int64)
           -> PlainOf target (TKScalar Int64) -> Int -> Int
           -> target (TKR 1 Int64)
      loop perm _ _ _ (-1) = perm
      loop perm elements idx fi i =
        let fi2 = fi `div` (i + 1)
            el = allFreeSpots elements ! [idx `quotH` fromIntegral fi2]
        in loop (updateR (len - i - 1) el perm)
                (updateI (tplainPart $ kfromR el) occupiedSpotMark elements)
                (idx `remH` fromIntegral fi2)
                fi2
                (i - 1)
  in loop perm0 elements0 idx0 fi0 (len - 1)

-- Compute the inversion number from a lexicographic index of a
-- permutation.
inversion_number_from_idx
  :: ADReady target
  => Int -> PlainOf target (TKScalar Int64) -> target (TKScalar Int64)
inversion_number_from_idx n idx0 =
  let loop s _ _ i | i == 0 = s
      loop s idx fi i =
        let fi2 = fi `div` (i + 1)
            s2 = s + idx `quotH` fromIntegral fi2
            idx2 = idx `remH` fromIntegral fi2
        in loop s2 idx2 fi2 (i - 1)
  in kfromPlain $ loop 0 idx0 (fact n) (n - 1)

productR :: ADReady target
         => target (TKR 1 Double) -> target (TKScalar Double)
productR = kfromR . rfold (*) (rscalar 1)

det :: forall target. ADReady target
    => target (TKR 2 Double) -> target (TKScalar Double)
det a =
  let ell = rwidth a
      f :: IntOf target -> target (TKScalar Double)
      f i =
        let p :: PlainOf target (TKR 1 Int64)
            p = tplainPart $ idx_to_perm @target ell i
            q :: target (TKScalar Double)
            q = kfromIntegral $ inversion_number_from_idx ell i
        in (-1) ** q
           * productR (rbuild1 ell $ \i2 -> a ! [i2, kfromR $ p ! [i2]])
  in kfromR $ rsum0 $ rbuild1 (fact ell) (rfromK . f)

primal :: Input -> PrimalOutput
primal (Input a ell) = unConcrete $ det $ chunk ell a

gradient :: Input -> GradientOutput
gradient (Input a ell) =
  Nested.rtoVector . unConcrete $ cgrad det (chunk ell a)
