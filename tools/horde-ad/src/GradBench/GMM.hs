{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module GradBench.GMM (objective, jacobian) where

import Control.DeepSeq (NFData (..))
import Data.Aeson (ToJSON (..), object, (.:), (.=))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Array.Nested.Ranked.Shape
import Data.Vector.Storable qualified as VS
import GHC.TypeLits (KnownNat)
import HordeAd
import HordeAd.Core.AstEnv
import HordeAd.Core.AstInterpret

foreign import ccall "lgamma" lgamma :: Double -> Double

type Independent target =
  ( target (TKR 1 Double)  -- gmmInAlphas :: [k]
  , target (TKR 2 Double)  -- gmmInMeans  :: [k][d]
  , target (TKR 2 Double)  -- gmmInQ      :: [k][d]
  , target (TKR 2 Double)  -- gmmInL      :: [k][1/2 d(d-1)]
  )

type role GMMIn representational
data GMMIn target =
  GMMIn { gmmIndependent :: Independent target
        , gmmInX         :: Concrete (TKR 2 Double)  -- [n][d]
        , gmmInWisGamma  :: Double
        , gmmInWisM      :: Int
        }

newtype GMMOut = GMMOut (Independent Concrete)

instance JSON.FromJSON (GMMIn Concrete) where
  parseJSON = JSON.withObject "input" $ \o -> do
    d <- o .: "d"
    k <- o .: "k"
    n <- o .: "n"
    alpha <- o .: "alpha"
    mu <- o .: "mu"
    q <- o .: "q"
    l <- o .: "l"
    x <- o .: "x"
    gamma <- o .: "gamma"
    m <- o .: "m"
    let gmmInAlphas = Concrete $ Nested.rfromVector [k] alpha
        gmmInMeans = chunk k d mu
        gmmInQ = chunk k d q
        gmmInL = chunk k (d * (d - 1) `div` 2) l
        gmmIndependent = (gmmInAlphas, gmmInMeans, gmmInQ, gmmInL)
        gmmInX = chunk n d x
        gmmInWisGamma = gamma
        gmmInWisM = m
    pure GMMIn{..}

instance JSON.ToJSON GMMOut where
  toJSON (GMMOut (gmmInAlphas, gmmInMeans, gmmInQ, gmmInL)) =
    object [ "alpha" .=
               JSON.toJSON (Nested.rtoVector $ unConcrete $ gmmInAlphas)
           , "mu" .= JSON.toJSON (unChunk gmmInMeans)
           , "q" .= JSON.toJSON (unChunk gmmInQ)
           , "l" .= JSON.toJSON (unChunk gmmInL)
           ]

chunk :: Int -> Int -> [VS.Vector Double] -> Concrete (TKR 2 Double)
chunk k d l =
  let xs = VS.concat l
  in Concrete $ Nested.rfromVector [k, d] xs

unChunk :: Concrete (TKR 2 Double) -> [VS.Vector Double]
unChunk = map (Nested.rtoVector . unConcrete) . runravelToList

instance NFData GMMOut where
  rnf (GMMOut (!_, _, _, _)) = ()

data Precomputed =
  Precomputed Double  -- ^ 1/2 N D log(2 pi)
              Double  -- ^ K (n' D (log gamma - 1/2 log(2))
                      --   - logGammaDistrib(1/2 n', D))

precompute :: ADReady target
           => GMMIn target -> Precomputed
precompute GMMIn{..} =
  let (gmmInAlphas, _, _, _) = gmmIndependent
      n :$: d :$: ZSR = rshape gmmInX
      k :$: ZSR = rshape gmmInAlphas
      n' = fromIntegral n
      d' = fromIntegral d
      k' = fromIntegral k
      n2 = d' + fromIntegral gmmInWisM + 1
  in Precomputed (0.5 * n' * d' * log (2 * pi))
                 (k' * (n2 * d' * (log gmmInWisGamma - 0.5 * log 2)
                        - logGammaDistrib (0.5 * n2) d))

logGammaDistrib :: Double -> Int -> Double
logGammaDistrib a d =
  fromIntegral (d * (d - 1)) * 0.25 * log pi
  + sum [lgamma (a + fromIntegral (1 - i) * 0.5) | i <- [1 .. d]]

frobeniusNormSq :: (KnownNat n, ADReady target)
                => target (TKR n Double) -> target (TKScalar Double)
frobeniusNormSq = rsum0 . rsquare

rlogsumexp' :: ADReady target
            => target (TKR 1 Double) -> target (TKScalar Double)
rlogsumexp' x = log (rsum0 (exp x))

logWishartPrior
  :: ADReady target
  => target (TKR 3 Double) -> target (TKR 1 Double)
  -> Double -> Int
  -> Precomputed
  -> target (TKScalar Double)
logWishartPrior qs sums wisGamma wisM (Precomputed _ c2) =
  kconcrete (fromIntegral wisM) * rsum0 sums
  - kconcrete (0.5 * wisGamma * wisGamma) * frobeniusNormSq qs
  + kconcrete c2

unpackQ :: ADReady target
        => Int -> target (TKR 1 Double) -> target (TKR 1 Double)
        -> target (TKR 2 Double)
unpackQ d logdiag lt =
  rbuild @2 [d, d] $ \[i, j] ->
    ifH (i <. j)
        (rscalar 0)
        (ifH (i <=. j)
             (exp (logdiag ! [i]))
             (lt ! [kconcrete (d - 1) * j + i - 1 - j * (j + 1) `quotH` 2]))

objectiveTarget :: ADReady target
                => GMMIn target -> target (TKScalar Double)
objectiveTarget input@GMMIn{..} =
  let prec@(Precomputed c1 _) = precompute input
      -- These four are variables (or projections of a variable),
      -- so they don't need to be shared via tlet.
      (gmmInAlphas, gmmInMeans, gmmInQ, gmmInL) = gmmIndependent
      n :$: d :$: ZSR = rshape gmmInX
      k :$: ZSR = rshape gmmInAlphas
      x = rconcrete $ unConcrete gmmInX
   in tlet (rzipWith1 (unpackQ d) gmmInQ gmmInL) $ \qs ->
      tlet (rsum $ rtr $ gmmInQ) $ \sums ->
      let innerTerm =
            rbuild1 n (\i ->
              rfromK $ rlogsumexp
                (gmmInAlphas + sums
                 - (rbuild1 k $ \j ->
                      let qximuk = rmatvecmul (qs ! [j])
                                              (x ! [i] - gmmInMeans ! [j])
                      in rfromK (0.5 * frobeniusNormSq qximuk))))
          slse = rsum0 innerTerm
      in slse
         - kconcrete (fromIntegral n) * rlogsumexp' gmmInAlphas
         + logWishartPrior qs sums gmmInWisGamma gmmInWisM prec
         - kconcrete c1

objectiveMixed :: ADReady target
               => GMMIn Concrete -> Independent target
               -> target (TKScalar Double)
objectiveMixed input gmmIndependent = objectiveTarget (input {gmmIndependent})

objective :: GMMIn Concrete -> Double
objective input =
  let (gmmInAlphas, gmmInMeans, gmmInQ, gmmInL) = gmmIndependent input
      concreteIndependent = ( rconcrete $ unConcrete gmmInAlphas
                            , rconcrete $ unConcrete gmmInMeans
                            , rconcrete $ unConcrete gmmInQ
                            , rconcrete $ unConcrete gmmInL
                            )
      ast = simplifyInlineContract $ objectiveMixed input concreteIndependent
  in -- traceShow ("primal", printAstPrettyButNested ast) $
     unConcrete $ interpretAstFull emptyEnv ast

jacobian :: GMMIn Concrete -> GMMOut
jacobian input = GMMOut $ grad (objectiveMixed input) (gmmIndependent input)
