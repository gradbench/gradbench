{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.Saddle (Input, Output, rr, ff, fr, rf) where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Vector.Storable qualified as VS
import GradBench.GD
import HordeAd

data Input = Input (Double, Double)

type Output = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "start"

fp :: (Floating a) => a -> a -> a -> a -> a
fp p1x p1y p2x p2y = (p1x ** 2 + p1y ** 2) - (p2x ** 2 + p2y ** 2)

saddleGen
  :: (NumScalar a, Differentiable a)
  => (Concrete (TKR 1 a) -> (Concrete (TKR 0 a), Concrete (TKR 1 a)))
  -> (Concrete (TKR 1 a) -> Concrete (TKR 1 a)
      -> (Concrete (TKR 0 a), Concrete (TKR 1 a)))
  -> Concrete (TKR 1 a)
  -> Concrete (TKR 1 a)
{-# INLINE saddleGen #-}
saddleGen r1cost' r2cost' start =
  let r1 = multivariateArgmin r1cost' start
      r2 = multivariateArgmax (r2cost' r1) start
  in rappend r1 r2

-- TODO: this is very slow for many reasons:
-- * nested concrete derivatives are slow, because they need to nest ADVal
-- * we can't use symbolic derivatives due to the recursion in multivariateMax
-- * we unroll all the identical things in the recursion and keep in memory
-- * nested derivatives in horde-ad are naively implemented regardless
-- * these are all 2-element rank 1 tensors (use products or lists instead?)
-- * fwd are slow, because they use Deltas instead of trivial dual numbers
-- * specialization and inlining is crucial here, but not investigated/forced
rr, ff, fr, rf :: Input -> Output
rr (Input (x, y)) = Nested.rtoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = rfromList [rscalar x, rscalar y]
    r1cost p1 = multivariateMax (r2cost' p1) (rfromPrimal start)
    r1cost' p1 = let (res0, res1) = cgrad2 (kfromR . r1cost) p1
                 in (rfromK res0, res1)
    r2cost :: forall target a. (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> target (TKR 1 a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (kfromR $ r1 ! [0]) (kfromR $ r1 ! [1])
                      (kfromR $ r2 ! [0]) (kfromR $ r2 ! [1])
    r2cost' :: forall target a.
               ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKR 1 a) -> target (TKR 1 a)
            -> (target (TKR 0 a), target (TKR 1 a))
    r2cost' r1 r2 =
      let (res0, res1) = cgrad2 @_ @_ @_ @target (r2cost (rfromPrimal r1)) r2
      in (rfromK res0, res1)
ff (Input (x, y)) = Nested.rtoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = rfromList [rscalar x, rscalar y]
    r1cost p1 = multivariateMax (r2cost' p1) (rfromPrimal start)
    r1cost' p1 = let (res0, res1) = cgrad2_fwdR (kfromR . r1cost) p1
                 in (rfromK res0, res1)
    r2cost :: forall target a. (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> target (TKR 1 a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (kfromR $ r1 ! [0]) (kfromR $ r1 ! [1])
                      (kfromR $ r2 ! [0]) (kfromR $ r2 ! [1])
    r2cost' :: forall target a.
               ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a, ADTensorScalar a ~ a )
            => target (TKR 1 a) -> target (TKR 1 a)
            -> (target (TKR 0 a), target (TKR 1 a))
    r2cost' r1 r2 =
      let (res0, res1) =
            cgrad2_fwdR @_ @_ @_ @target (r2cost (rfromPrimal r1)) r2
      in (rfromK res0, res1)
rf (Input (x, y)) = Nested.rtoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = rfromList [rscalar x, rscalar y]
    r1cost p1 = multivariateMax (r2cost' p1) (rfromPrimal start)
    r1cost' p1 = let (res0, res1) = cgrad2 (kfromR . r1cost) p1
                 in (rfromK res0, res1)
    r2cost :: forall target a. (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> target (TKR 1 a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (kfromR $ r1 ! [0]) (kfromR $ r1 ! [1])
                      (kfromR $ r2 ! [0]) (kfromR $ r2 ! [1])
    r2cost' :: forall target a.
               ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a, ADTensorScalar a ~ a )
            => target (TKR 1 a) -> target (TKR 1 a)
            -> (target (TKR 0 a), target (TKR 1 a))
    r2cost' r1 r2 =
      let (res0, res1) =
            cgrad2_fwdR @_ @_ @_ @target (r2cost (rfromPrimal r1)) r2
      in (rfromK res0, res1)
fr (Input (x, y)) = Nested.rtoVector . unConcrete
                    $ saddleGen r1cost' r2cost' start
  where
    start = rfromList [rscalar x, rscalar y]
    r1cost p1 = multivariateMax (r2cost' p1) (rfromPrimal start)
    r1cost' p1 = let (res0, res1) = cgrad2_fwdR (kfromR . r1cost) p1
                 in (rfromK res0, res1)
    r2cost :: forall target a. (NumScalar a, Differentiable a, ADReady target)
           => target (TKR 1 a) -> target (TKR 1 a)
           -> target (TKScalar a)
    r2cost r1 r2 = fp (kfromR $ r1 ! [0]) (kfromR $ r1 ! [1])
                      (kfromR $ r2 ! [0]) (kfromR $ r2 ! [1])
    r2cost' :: forall target a.
               ( ADReadyNoLet target, ShareTensor target
               , ShareTensor (PrimalOf target), ShareTensor (PlainOf target)
               , NumScalar a, Differentiable a )
            => target (TKR 1 a) -> target (TKR 1 a)
            -> (target (TKR 0 a), target (TKR 1 a))
    r2cost' r1 r2 =
      let (res0, res1) = cgrad2 @_ @_ @_ @target (r2cost (rfromPrimal r1)) r2
      in (rfromK res0, res1)
