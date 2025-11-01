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
  let dummyEl :: forall f. ADReady f => f (TKR 0 Int64)
      dummyEl = rscalar (-1)
      occupiedSpotMark :: forall f. ADReady f => f (TKR 0 Int64)
      occupiedSpotMark = rscalar (-1)
      perm0 = rreplicate len dummyEl
      elements0 = riota len
      fi0 = fact len
  in withSNat len $ \k ->
    let allFreeSpots :: forall g. ADReady g
                     => g (TKR 1 Int64) -> g (TKR 1 Int64)
        allFreeSpots elements1 =
          let f :: forall f. ADReady f
                => f (TKProduct (TKR 1 Int64) (TKScalar Int64))
                -> f (TKR 0 Int64)
                -> f (TKProduct (TKR 1 Int64) (TKScalar Int64))
              f acc el =  -- sharing not needed, because these are variables
                ifH (el ==. occupiedSpotMark)
                    acc
                    (tpair
                       (updateI (kplainPart $ tproject2 acc) el (tproject1 acc))
                       (tproject2 acc + 1))
              acc0 = tpair (rreplicate len dummyEl) 0
          in tproject1 $ tfold k knownSTK knownSTK f acc0 elements1
        mkPerm :: target (TKR 1 Int64) -> target (TKR 1 Int64)
               -> target (TKScalar Int64) -> target (TKScalar Int64)
               -> target (TKR 1 Int64)
        mkPerm perm1 elements1 idx1 fi1 =
          let f :: forall f. ADReady f
                => f (TKProduct (TKR 1 Int64)
                                (TKProduct (TKR 1 Int64)
                                           (TKProduct (TKScalar Int64)
                                                      (TKScalar Int64))))
                -> f (TKR 0 Int64)
                -> f (TKProduct (TKR 1 Int64)
                                (TKProduct (TKR 1 Int64)
                                           (TKProduct (TKScalar Int64)
                                                      (TKScalar Int64))))
              f acc i =  -- sharing not needed, because these are variables
                let perm = tproject1 acc
                    elements' = tproject1 (tproject2 acc)
                    idx' = tproject1 (tproject2 (tproject2 acc))
                    fi' = tproject2 (tproject2 (tproject2 acc))
                in tlet elements' $ \elements ->
                   tlet idx' $ \idx ->
                   tlet (fi' `quotH` (kfromR i + 1)) $ \fi2 ->
                   tlet (allFreeSpots elements
                         ! [kplainPart $ idx `quotH` fi2]) $ \el ->
                     let perm2 = updateI (fromIntegral len
                                          - kplainPart (kfromR i) - 1)
                                         el perm
                         elements2 = updateI (kplainPart $ kfromR el)
                                             occupiedSpotMark elements
                         idx2 = idx `remH` fi2
                     in tpair perm2 (tpair elements2 (tpair idx2 fi2))
              acc0 = tpair perm1 (tpair elements1 (tpair idx1 fi1))
          in tproject1 $ tfold k knownSTK knownSTK f acc0 (rreverse $ riota len)
    in mkPerm perm0 elements0 (kfromPlain idx0) (fromIntegral fi0)

-- Compute the inversion number from a lexicographic index of a
-- permutation.
-- This is not computed in @PlainOf target@ but, in theory, when given
-- an argument of the form @kfromPlain i@, it should reduce to a term
-- of the form @kfromPlain result@ already when first converted to an AST term.
inversion_number_from_idx
  :: forall target. ADReady target
  => Int -> target (TKScalar Int64) -> target (TKScalar Int64)
inversion_number_from_idx len idx0 =
  let f :: forall f. ADReady f
        => f (TKProduct (TKScalar Int64)
                        (TKProduct (TKScalar Int64)
                                   (TKScalar Int64)))
        -> f (TKR 0 Int64)
        -> f (TKProduct (TKScalar Int64)
                        (TKProduct (TKScalar Int64)
                                   (TKScalar Int64)))
      f acc i =  -- sharing not needed, because these are variables
        let s = tproject1 acc
            idx' = tproject1 (tproject2 acc)
            fi' = tproject2 (tproject2 acc)
        in tlet idx' $ \idx ->
           tlet (fi' `quotH` (kfromR i + 2)) $ \fi2 ->
             let s2 = s + idx `quotH` fi2
                 idx2 = idx `remH` fi2
             in tpair s2 (tpair idx2 fi2)
      acc0 = tpair 0 (tpair idx0 (fromIntegral $ fact len))
  in withSNat (len - 1) $ \k1 ->
       tproject1 $ tfold k1 knownSTK knownSTK f acc0
                         (rreverse $ riota (len - 1))

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
              -- Ideally the delta-term discarded by this tplainPart should be
              -- DeltaZero, but is our AST rewriting that good already?
            q :: target (TKScalar Double)
            q = kfromIntegral
                $ inversion_number_from_idx @target ell (kfromPlain i)
        in (-1) ** q
           * productR (rgather1 ell a $ \i2 -> [i2, kfromR $ p ! [i2]])
  in kfromR $ rsum0 $ rbuild1 (fact ell) (rfromK . f)

primal :: Input -> PrimalOutput
primal (Input a ell) = unConcrete $ det $ chunk ell a

gradient :: Input -> GradientOutput
gradient (Input a ell) =
  Nested.rtoVector . unConcrete $ grad det (chunk ell a)
