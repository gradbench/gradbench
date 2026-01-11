{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.KMeans
  ( cost,
    dir,
    Input,
    CostOutput,
    DirOutput,
  )
where

import Control.DeepSeq (NFData (..))
import Data.Aeson (ToJSON (..), (.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.List.NonEmpty qualified as NE
import Data.Vector.Storable qualified as VS
import HordeAd

getPoint :: VS.Storable a => Int -> VS.Vector a -> Int -> VS.Vector a
getPoint d v i = VS.slice (i * d) d v

getPoints :: VS.Storable a => Int -> VS.Vector a -> [VS.Vector a]
getPoints d v =
  map (getPoint d v) [0 .. (VS.length v `div` d - 1)]

data Input = Input
  { inputD :: Int,
    inputPoints :: VS.Vector Double,
    inputCentroids :: VS.Vector Double
  }

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o -> do
    points <- o .: "points"
    centroids <- o .: "centroids"
    pure $
      Input
        { inputD = VS.length $ NE.head points,
          inputPoints = VS.concat $ NE.toList points,
          inputCentroids = VS.concat $ NE.toList centroids
        }

type CostOutput = Double

data DirOutput = DirOutput Int (VS.Vector Double)

instance JSON.ToJSON DirOutput where
  toJSON (DirOutput d v) = JSON.toJSON $ getPoints d v

instance NFData DirOutput where
  rnf (DirOutput _ v) = rnf v

square :: (NumScalar a, ADReady target)
       => target (TKR 1 a) -> target (TKR 1 a)
square x' = tlet x' $ \x -> x * x
-- slower even symbolically: square x = x ** rrepl (rshape x) 2

dist2 :: (NumScalar a, ADReady target)
      => target (TKR 1 a) -> target (TKR 1 a) -> target (TKR 0 a)
dist2 a b = rsum0 $ square $ a - b

-- TODO: try fold instead of build
costGeneric :: (NumScalar a, ADReady target)
            => target (TKR 2 a) -> target (TKR 2 a) -> target (TKScalar a)
costGeneric points centroids =
  kfromR $ rsum0
  $ rbuild1 (rwidth points)
            (\ip -> rminimum
                    $ rbuild1 (rwidth centroids)
                              (\ic -> dist2 (points ! [ip]) (centroids ! [ic])))

cost :: Input -> CostOutput
cost (Input d points' centroids') =
  let points =
        rconcrete
        $ Nested.rfromVector [VS.length points' `div` d, d] points'
      csh = [VS.length centroids' `div` d, d]
      centroids = rconcrete $ Nested.rfromVector csh centroids'
  in unConcrete $ costGeneric points centroids

dir :: Input -> DirOutput
dir (Input d points' centroids') =
  let points :: ADReady target => target (TKR 2 Double)
      points =
        rconcrete $ Nested.rfromVector [VS.length points' `div` d, d] points'
      csh = [VS.length centroids' `div` d, d]
      centroids = rconcrete $ Nested.rfromVector csh centroids'
      (cost', cost'') = jvp2 (kgrad (costGeneric points) (FTKR csh FTKScalar))
                             centroids
                             (rrepl (rshape centroids) 1)
  in DirOutput d . Nested.rtoVector . unConcrete $ cost' / cost''
    -- non-symbolic cjvp2 would take much more memory and time here
    -- due to rbuild1 above
