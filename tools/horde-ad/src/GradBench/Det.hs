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

import Control.Monad.ST.Strict (ST, runST)
import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Int (Int64, Int8)
import Data.Vector.Storable qualified as VS
import Data.Vector.Storable.Mutable qualified as VSM
import HordeAd
import HordeAd.Core.AstEnv
import HordeAd.Core.AstInterpret

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

fused :: forall s.
         Int -> Int -> VSM.MVector s Int64 -> VSM.MVector s Bool -> ST s ()
fused !len !idx0 !perm !freeSpots = do
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

mutated :: forall s. Int -> ST s (VS.Vector Int64)
mutated !len = do
  perms <- VSM.unsafeNew (len * fact len)
  freeSpots <- VSM.unsafeNew len
  let loop :: Int -> ST s ()
      loop (-1) = return ()
      loop i = do
        VSM.set freeSpots True
        fused len i (VSM.slice (i * len) len perms) freeSpots
        loop (i - 1)
  loop (fact len - 1)
  VS.unsafeFreeze perms

-- Given the lexicographic index of a permutation, compute that
-- permutation.
idx_to_perm :: Int -> Nested.Ranked 2 Int64
idx_to_perm n = Nested.rfromVector [fact n, n] $ runST $ mutated n

-- Compute the inversion number from a lexicographic index of a
-- permutation.
inversion_number_from_idx :: Int -> Nested.Ranked 1 Int8
inversion_number_from_idx n =
  let loop s _ _ i | i == 1 = fromIntegral s
      loop s idx fi i =
        let fi2 = fi `quot` i
            (s1, idx2) = idx `quotRem` fi2
            s2 = s + s1
        in loop s2 idx2 fi2 (i - 1)
      f idx0 = loop 0 idx0 (fact n) n
  in Nested.rfromVector [fact n] $ VS.generate (fact n) f

productR :: ADReady target
         => target (TKR 1 Double) -> target (TKScalar Double)
productR = kfromR . rfold (*) (rscalar 1)

det :: forall target. ADReady target
    => target (TKR 2 Double) -> target (TKScalar Double)
det a =
  let ell = rwidth a
      p :: PlainOf target (TKR 2 Int64)
      p = rconcrete $ idx_to_perm ell
      q :: PlainOf target (TKR 1 Int8)
      q = rconcrete $ inversion_number_from_idx ell
      f :: IntOf target -> target (TKScalar Double)
      f i = (-1) ** kfromPlain (kfromIntegral (q `rindex0` [i]))
            * productR (rgather1 ell a $ \i2 ->
                          [i2, kfromIntegral $ p `rindex0` [i, i2]])
  in withSNat (fact ell) $ \ (SNat @k) ->
       ssum0 $ kbuild1 @k f

primal :: Input -> PrimalOutput
primal (Input a ell) =
  let ast = simplifyInlineContract $ tlet (chunk ell a) det
  in -- traceShow ("pre", printAstPrettyButNested ast) $
     unConcrete $ interpretAstFull emptyEnv ast

gradient :: Input -> GradientOutput
gradient (Input a ell) =
  Nested.rtoVector . unConcrete $ grad det (chunk ell a)
