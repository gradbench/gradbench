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
import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Array.Nested.Ranked.Shape
import Data.Int (Int64)
import Data.Vector.Storable qualified as VS
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

listUpdate :: Int -> (a -> a) -> [a] -> [a]
listUpdate 0 f (x : xs) = f x : xs
listUpdate i f (x : xs) = x : listUpdate (i - 1) f xs
listUpdate _ _ [] = error "listUpdate: index too large"

fused :: Int -> Int -> [Int64]
fused len idx0 =
  let perm0 = replicate len (-1)
      elements0 = replicate len False
      fi0 = fact len
      nthFreeSpot :: [Bool] -> Int -> Int64 -> Int64
      nthFreeSpot elements1 n el =
        let free = not (elements1 !! fromIntegral el)
        in if n <= 0 && free
           then el
           else nthFreeSpot elements1 (n - fromEnum free) (el + 1)
      loop :: [Int64] -> [Bool] -> Int -> Int -> Int -> [Int64]
      loop perm _ _ _ 0 = perm
      loop perm elements idx fi i2 =
        let fi2 = fi `quot` i2
            (idxDigit, idxRest) = idx `quotRem` fi2
            el = nthFreeSpot elements idxDigit 0
        in loop (listUpdate (len - i2) (const el) perm )
                (listUpdate (fromIntegral el) (const True) elements)
                idxRest
                fi2
                (i2 - 1)
  in loop perm0 elements0 idx0 fi0 len

-- Given the lexicographic index of a permutation, compute that
-- permutation.
idx_to_perm :: Int -> Nested.Ranked 2 Int64
idx_to_perm n =
  let f idx0 = Nested.rfromListPrim $ fused n idx0
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
