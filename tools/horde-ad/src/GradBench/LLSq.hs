{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module GradBench.LLSq
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
import Data.Array.Nested.Shaped.Shape
import Data.Vector.Storable qualified as VS
import GHC.TypeLits (KnownNat, type (+))
import HordeAd
import HordeAd.Core.AstEnv
import HordeAd.Core.AstInterpret

data Input = Input
  { _inputX :: VS.Vector Double,
    _inputN :: Int
  }

type PrimalOutput = Double

type GradientOutput = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (o .: "x") <*> o .: "n"

square :: (NumScalar a, ADReady target)
       => target (TKScalar a) -> target (TKScalar a)
square x' = tlet x' $ \x -> x * x
  -- slower even symbolically: square x = x ** rrepl (rshape x) 2

primalPoly :: forall nxm1 a target.
              (KnownNat nxm1, NumScalar a, Differentiable a, ADReady target )
           => Int -> target (TKS '[nxm1 + 1] a) -> target (TKScalar a)
primalPoly n x =
  let f i = tletPlain (t i n) $ \ti ->
        let muls :: PlainOf target (TKS '[nxm1 + 1] a)
            muls = tscan (SNat @nxm1) STKScalar STKScalar
                         (*) 1 $ sreplicate0N @'[nxm1] ti
        in square (kfromPlain (signum ti)
                   - ssum0 (x * tfromPlain
                                  (STKS (SNat @(nxm1 + 1) :$$ ZSS) STKScalar)
                                  muls))
  in withSNat n $ \(SNat @n) ->
       0.5 * ssum0 (kbuild1 @n f)
 where
  t :: IntOf target -> Int -> PlainOf target (TKScalar a)
  t i k = negate 1 + (kfromIntegral i) * 2 / (fromIntegral k - 1)

{- this variant is much slower:
t :: (NumScalar a, Differentiable a, ADReady target)
  => IntOf target -> Int -> target (TKScalar a)
t i n = negate 1 + kfromPlain (kfromIntegral i) * 2 / (fromIntegral n - 1)

primalPoly :: forall nxm1 a target.
              (KnownNat nxm1, NumScalar a, Differentiable a, ADReady target )
           => target (TKS '[nxm1 + 1] a) -> Int -> target (TKScalar a)
primalPoly x n =
  withSNat n $ \(SNat @n) ->
  let f i = tlet (t i n) $ \ti ->
        let muls :: target (TKProduct (TKProduct (TKScalar a) (TKScalar a))
                                      (TKScalar a))
            muls = tfold (SNat @(nxm1 + 1))
                         (STKProduct (STKProduct STKScalar STKScalar)
                                     STKScalar)
                         STKScalar
                         (\ti_tii_acc' ex -> tlet ti_tii_acc' $ \ti_tii_acc ->
                            tpair (tpair (tproject1 (tproject1 ti_tii_acc))
                                         (tproject1 (tproject1 ti_tii_acc)
                                          * tproject2 (tproject1 ti_tii_acc)))
                                  (tproject2 ti_tii_acc
                                   + tproject2 (tproject1 ti_tii_acc) * ex))
                         (tpair (tpair ti 1) 0) x
        in square (signum ti - tproject2 muls)
  in 0.5 * ssum0 (kbuild1 @n f) -}

primal :: Input -> PrimalOutput
primal (Input x n) =
  withSNat (VS.length x - 1) $ \(SNat @nxm1) ->
  let y = sconcrete . Nested.sfromVector (SNat @(nxm1 + 1) :$$ ZSS) $ x
      ast = simplifyInlineContract $ primalPoly n y
  in -- traceShow ("primal", printAstPrettyButNested ast) $
     unConcrete $ interpretAstFull emptyEnv ast
       -- vectorizations speeds this up orders of magnitude

gradient :: Input -> GradientOutput
gradient (Input x n) =
  withSNat (VS.length x - 1) $ \(SNat @nxm1) ->
  let y = sconcrete . Nested.sfromVector (SNat @(nxm1 + 1) :$$ ZSS) $ x
  in Nested.stoVector . unConcrete $ grad (primalPoly n) y
    -- non-symbolic cgrad would take much more memory and time here
    -- due to rbuild1 above
