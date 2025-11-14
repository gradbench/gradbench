{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
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
import Control.Monad.ST.Strict (ST, runST)
import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Array.Nested.Ranked.Shape
import Data.Int (Int64)
import Data.Vector.Storable qualified as VS
import Data.Vector.Storable.Mutable qualified as VSM
import Foreign.C (CInt)
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
chunk n xs = rconcrete $ Nested.rfromVector [VS.length xs `quot` n, n] xs

fact :: Int -> Int
fact n = factAcc 1 n
 where factAcc acc i | i <= 1 = acc
       factAcc acc i = factAcc (i * acc) (i - 1)

fused :: forall s. Int -> Int -> ST s (VS.Vector Int64)
fused !len !idx0 = do
  perm <- VSM.replicate len (-1)
  freeSpots <- VSM.replicate len True
  let nthFreeSpot :: Int -> Int -> ST s Int
      nthFreeSpot !pos !el = do
        free <- VSM.read freeSpots el
        if pos <= 0 && free
        then return el
        else nthFreeSpot (pos - fromEnum free) (el + 1)
      loop :: Int -> Int -> Int -> ST s ()
      loop _ _ 0 = return ()
      loop !idx !fi i2 = do
        let fi2 = fi `quot` i2
            (idxDigit, idxRest) = idx `quotRem` fi2
        el <- nthFreeSpot idxDigit 0
        VSM.write perm (len - i2) (fromIntegral el)
        VSM.write freeSpots el False
        loop idxRest fi2 (i2 - 1)
  loop idx0 (fact len) len
  VS.unsafeFreeze perm

-- Given the lexicographic index of a permutation, compute that
-- permutation.
idx_to_perm :: Int -> Nested.Ranked 2 Int64
idx_to_perm n =
  let f idx0 = Nested.rfromVector [n] $ runST $ fused n idx0
  in tbuild1R (fact n) f

tbuild1R
  :: forall r n. (Nested.KnownElt r, KnownNat n)
  => Int -> (Int -> Nested.Ranked n r) -> Nested.Ranked (1 + n) r
tbuild1R k f =
  Nested.runNest $ Nested.rgenerate (k :$: ZSR) $ \(i :.: ZIR) -> f i

-- Compute the inversion number from a lexicographic index of a
-- permutation.
inversion_number_from_idx :: Int -> Nested.Ranked 1 CInt
inversion_number_from_idx n =
  let loop s _ _ i | i == 1 = s
      loop s idx fi i =
        let fi2 = fi `quot` i
            (s1, idx2) = idx `quotRem` fi2
            s2 = s + s1
        in loop s2 idx2 fi2 (i - 1)
      f (idx0 :.: ZIR) =
        loop 0 (fromIntegral idx0) (fromIntegral $ fact n) (fromIntegral n)
  in Nested.rgenerate [fact n] f

productR :: ADReady target
         => target (TKR 1 Double) -> target (TKScalar Double)
productR = kfromR . rfold (*) (rscalar 1)

det :: forall target. ADReady target
    => target (TKR 2 Double) -> target (TKScalar Double)
det a =
  let ell = rwidth a
      p :: PlainOf target (TKR 2 Int64)
      p = rconcrete $ idx_to_perm ell
      q :: PlainOf target (TKR 1 CInt)
      q = rconcrete $ inversion_number_from_idx ell
      f :: IntOf target -> target (TKScalar Double)
      f i = (-1) ** kfromPlain (kfromIntegral (q `rindex0` [i]))
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
