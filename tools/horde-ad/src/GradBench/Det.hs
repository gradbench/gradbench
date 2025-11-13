{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
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

import Control.Concurrent
import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Int (Int64)
import Data.Vector.Storable qualified as VS
import GHC.TypeLits (KnownNat, type (+))
import HordeAd
import HordeAd.Core.AstEnv
import HordeAd.Core.AstInterpret
import System.IO.Unsafe (unsafePerformIO)

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

updateR :: (GoodScalar r, ADReady target)
        => IntOf target -> target (TKScalar r) -> target (TKR 1 r)
        -> target (TKR 1 r)
updateR idx a v =
  rgather1 (rwidth v) (v `rappend` rreplicate 1 (rfromK a)) $ \i ->
    [ifH (i ==. idx) (fromIntegral $ rwidth v) i]

updateS :: forall k r target. (KnownNat k, GoodScalar r, ADReady target)
        => IntOf target -> target (TKScalar r) -> target (TKS '[k] r)
        -> target (TKS '[k] r)
updateS idx a v =
  sgather1 @_ @_ @'[k + 1] (v `sappend` sreplicate @1 (sfromK a)) $ \i ->
    [ifH (i ==. idx) (fromIntegral $ swidth v) i]

-- Given the lexicographic index of a permutation, compute that
-- permutation.
idx_to_perm :: forall target. (BaseTensor target, LetTensor target)
            => Int -> target (TKR 2 Int64)
idx_to_perm len = withSNat len $ \(k :: SNat k) ->
  let dummyEl :: forall f. BaseTensor f => f (TKR 0 Int64)
      dummyEl = rscalar (-1)
      occupiedSpotMark :: forall f. BaseTensor f => f (TKScalar Int64)
      occupiedSpotMark = -1
      perm0 = rreplicate len dummyEl
      fi0 = fact len
      elements0 = siota
      allFreeSpots :: forall g. ADReady g
                   => g (TKS '[k] Int64) -> g (TKR 1 Int64)
      allFreeSpots elements1 =
        let f :: forall f. ADReady f
              => f (TKProduct (TKR 1 Int64) (TKScalar Int64))
              -> f (TKScalar Int64)
              -> f (TKProduct (TKR 1 Int64) (TKScalar Int64))
            f acc el =  -- sharing not needed, because these are variables
              let spots = tproject1 acc
                  nOfFree = tproject2 acc
              in ifH (el <=. occupiedSpotMark)
                     acc
                     (tpair (updateR (kplainPart nOfFree) el spots)
                            (nOfFree + 1))
            acc0 = tpair (rreplicate len dummyEl) 0
        in tproject1 $ tfold k knownSTK knownSTK f acc0 elements1
      mkPerm :: target (TKR 1 Int64) -> target (TKS '[k] Int64)
             -> target (TKScalar Int64) -> target (TKScalar Int64)
             -> target (TKR 1 Int64)
      mkPerm perm1 elements1 idx1 fi1 =
        let f :: forall f. ADReady f
              => f (TKProduct (TKR 1 Int64)
                              (TKProduct (TKS '[k] Int64)
                                         (TKProduct (TKScalar Int64)
                                                    (TKScalar Int64))))
              -> f (TKScalar Int64)
              -> f (TKProduct (TKR 1 Int64)
                              (TKProduct (TKS '[k] Int64)
                                         (TKProduct (TKScalar Int64)
                                                    (TKScalar Int64))))
            f acc i1 =  -- sharing not needed, because these are variables
              let perm = tproject1 acc  -- not tlet: projections not shared
                  elements = tproject1 (tproject2 acc)
                  idx = tproject1 (tproject2 (tproject2 acc))
                  fi = tproject2 (tproject2 (tproject2 acc))
              in tlet (fi `quotH` i1) $ \fi2 ->
                 tlet (allFreeSpots elements
                       `rindex0` [kplainPart $ idx `quotH` fi2]) $ \el ->
                   let perm2 = updateR (fromIntegral len - kplainPart i1)
                                       el perm
                       elements2 = updateS (kplainPart el)
                                           occupiedSpotMark elements
                       idx2 = idx `remH` fi2
                   in tpair perm2 (tpair elements2 (tpair idx2 fi2))
            acc0 = tpair perm1 (tpair elements1 (tpair idx1 fi1))
            l = sslice (SNat @0) SNat (SNat @1) $ sreverse siota
        in tproject1 $ tfold k knownSTK knownSTK f acc0 l
  in rbuild1 fi0 $ \idx0 ->
       mkPerm perm0 elements0 (kfromPlain idx0) (fromIntegral fi0)

-- Compute the inversion number from a lexicographic index of a
-- permutation.
-- This is not computed in @PlainOf target@ but @kfromIntegral@
-- applies @astPlainPart@ to it, which causes it to quickly reduce to a term
-- in @PlainOf target@.
inversion_number_from_idx
  :: forall target. (BaseTensor target, LetTensor target, ConvertTensor target)
  => Int -> target (TKR 1 Int64)
inversion_number_from_idx len = withSNat (len - 1) $ \k1 ->
  let fi0 = fact len
      f :: forall f. ADReady f
        => f (TKProduct (TKScalar Int64)
                        (TKProduct (TKScalar Int64)
                                   (TKScalar Int64)))
        -> f (TKScalar Int64)
        -> f (TKProduct (TKScalar Int64)
                        (TKProduct (TKScalar Int64)
                                   (TKScalar Int64)))
      f acc i2 =  -- sharing not needed, because these are variables
        let s = tproject1 acc
            idx = tproject1 (tproject2 acc)
            fi = tproject2 (tproject2 acc)
        in tlet (fi `quotH` i2) $ \fi2 ->
             let s2 = s + idx `quotH` fi2
                 idx2 = idx `remH` fi2
             in tpair s2 (tpair idx2 fi2)
      l = sslice (SNat @0) SNat (SNat @2) $ sreverse siota
  in rbuild1 fi0 $ \idx0 ->
       let acc0 = tpair 0 (tpair (kfromPlain idx0) (fromIntegral fi0))
       in rfromK $ tproject1 $ tfold k1 knownSTK knownSTK f acc0 l

productR :: ADReady target
         => target (TKR 1 Double) -> target (TKScalar Double)
productR = kfromR . rfold (*) (rscalar 1)

det :: forall target. ADReady target
    => target (TKR 2 Double) -> target (TKScalar Double)
det a =
  let ell = rwidth a
      p :: PlainOf target (TKR 2 Int64)
      p = rconcrete $ unConcrete $ idx_to_perm ell
      q :: PlainOf target (TKR 1 Double)
      q = rconcrete $ unConcrete $ rfromIntegral
          $ inversion_number_from_idx ell
      f :: IntOf target -> target (TKScalar Double)
      f i = (-1) ** kfromPlain (q `rindex0` [i])
            * productR (rgather1 ell a $ \i2 -> [i2, p `rindex0` [i, i2]])
  in kfromR $ rsum $ rbuild1 (fact ell) (rfromK . f)

primal :: Input -> PrimalOutput
primal (Input a ell) =
  let ast = simplifyInlineContract $ tlet (chunk ell a) det
  in -- traceShow ("pre", printAstPrettyButNested ast) $
     unConcrete $ interpretAstFull emptyEnv ast

gradient :: Input -> GradientOutput
gradient (Input a ell) =
  Nested.rtoVector . unConcrete $ grad det (chunk ell a)
