{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.Saddle (Input, Output, rr, ff, fr, rf) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Array.Nested.Lemmas
import Data.Array.Nested.Shaped.Shape
import Data.Type.Equality ((:~:) (Refl))
import Data.Vector.Storable qualified as VS
import GradBench.GD
import HordeAd
import HordeAd.Core.Adaptor

newtype Input = Input (Double, Double)

type Output = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "start"

magnitude_squaredS :: (NumScalar a, ADReady target)
                   => target (TKS '[2] a) -> target (TKScalar a)
{-# INLINE magnitude_squaredS #-}
magnitude_squaredS t = t `sindex0` [0] * t `sindex0` [0]
                       + t `sindex0` [1] * t `sindex0` [1]

scaleS :: (NumScalar a, ADReady target)
       => a -> target (TKS '[2] a) -> target (TKS '[2] a)
{-# INLINE scaleS #-}
scaleS x v = sconcrete (Nested.sreplicatePrim (SNat @2 :$$ ZSS) x) * v

cgrad2_fwdS
  :: forall src r tgt target sh.
     ( src ~ ADVal target (TKS sh r)
     , NumScalar r, ADTensorScalar r ~ r, KnownShS sh
     , tgt ~ ADVal target (TKScalar r)
     , ADReadyNoLet target, ShareTensor target
     , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
  => (src -> tgt)  -- ^ the objective function
  -> DValue src
  -> ( target (TKScalar r)
     , DValue src )  -- morally DValue (ADTensorKind src)
{-# INLINE cgrad2_fwdS #-}
cgrad2_fwdS f x | Refl <- lemAppNil @sh =
  let g :: IxSOf target sh -> target (TKScalar r)
      g i = cjvp f x (soneHot (sscalar 1) i)
  in (kprimalPart $ f (fromDValue x), kbuild g)
{- The following is slower, probably because kbuild has an efficient
   implementation in the Concrete instance (one of the two instances
   where this is used). We could define this differently for each instance,
   but that would be too ugly even for a benchmark code.
cgrad2_fwdS f x =
  let v10 = singestData [1, 0]
      v01 = singestData [0, 1]
  in ( kprimalPart $ f (fromDValue x)
     , sfromVectorLinear (SNat @2 :$$ ZSS)
       $ V.fromList [cjvp f x v10, cjvp f x v01] )
-}

fp :: Floating a => a -> a -> a -> a -> a
{-# INLINE fp #-}
fp p1x p1y p2x p2y = (p1x ** 2 + p1y ** 2) - (p2x ** 2 + p2y ** 2)
  -- this is slightly slower:
  -- fp p1x p1y p2x p2y = (p1x * p1x + p1y * p1y) - (p2x * p2x + p2y * p2y)

saddleGen
  :: forall a. a ~ Double
  => (forall target.
      ( ADReadyNoLet target, ShareTensor target
      , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
      => (ADVal target (TKS '[2] a) -> ADVal target (TKScalar a))
      -> target (TKS '[2] a)
      -> (target (TKScalar a), target (TKS '[2] a)))
  -> (forall target.
      ( ADReadyNoLet target, ShareTensor target
      , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
      => (ADVal target (TKS '[2] a) -> ADVal target (TKScalar a))
      -> target (TKS '[2] a)
      -> (target (TKScalar a), target (TKS '[2] a)))
  -> Input
  -> Output
{-# INLINE saddleGen #-}
saddleGen cgrad2A cgrad2B (Input (x, y)) =
  let start = sfromListLinear (SNat @2 :$$ ZSS) [x, y]
      r1cost :: ADVal Concrete (TKS '[2] a) -> ADVal Concrete (TKScalar a)
      r1cost p1 = multivariateMax magnitude_squaredS scaleS
                                  (r2cost' p1) (sfromPrimal start)
      r1cost' :: Concrete (TKS '[2] a)
              -> (Concrete (TKScalar a), Concrete (TKS '[2] a))
      r1cost' = cgrad2A r1cost
      r2cost :: BaseTensor target
             => target (TKS '[2] a) -> target (TKS '[2] a)
             -> target (TKScalar a)
      r2cost r1 r2 = fp (r1 `sindex0` [0]) (r1 `sindex0` [1])
                        (r2 `sindex0` [0]) (r2 `sindex0` [1])
      -- This is slower most of the time:
      -- r2cost r1 r2 = sdot0 r1 r1 - sdot0 r2 r2
      r2cost' :: ( ADReadyNoLet target, ShareTensor target
                 , ShareTensor (PrimalOf target), ShareTensor (PlainOf target) )
              => target (TKS '[2] a) -> target (TKS '[2] a)
              -> (target (TKScalar a), target (TKS '[2] a))
      r2cost' r1 = cgrad2B (r2cost (sfromPrimal r1))
      res1 = multivariateArgmin magnitude_squaredS scaleS r1cost' start
      res2 = multivariateArgmax magnitude_squaredS scaleS (r2cost' res1) start
  in Nested.stoVector $ unConcrete $ sappend res1 res2

-- TODO: this is very slow for many reasons:
-- * nested concrete derivatives are slow, because they need to nest ADVal
-- * we can't use symbolic derivatives due to the non-structured recursion
--   in multivariateMax
-- * we unroll all the identical things in the recursion and keep in memory
-- * nested derivatives in horde-ad are naively implemented
-- * these are all 2-element rank 1 tensors, so a lot of meta-data overhead
-- * fwd are slow, because they use Deltas instead of trivial dual numbers
rr, ff, fr, rf :: Input -> Output
rr = saddleGen cgrad2 cgrad2
ff = saddleGen cgrad2_fwdS cgrad2_fwdS
rf = saddleGen cgrad2 cgrad2_fwdS
fr = saddleGen cgrad2_fwdS cgrad2
