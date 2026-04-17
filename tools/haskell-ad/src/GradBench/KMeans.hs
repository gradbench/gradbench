module GradBench.KMeans
  ( cost,
    dir,
    Input,
    CostOutput,
    DirOutput,
  )
where

import           Control.DeepSeq    (NFData (..))
import           Data.Aeson         (ToJSON (..), (.:))
import qualified Data.Aeson         as JSON
import qualified Data.List          as L
import qualified Data.List.NonEmpty as NE
import qualified Data.Vector        as V
import qualified Numeric.AD.Double  as D

getPoint :: Int -> V.Vector a -> Int -> [a]
getPoint d v i = V.toList $ V.slice (i * d) d v

getPoints :: Int -> V.Vector a -> [[a]]
getPoints d v =
  map (getPoint d v) [0 .. (V.length v `div` d - 1)]

data Input = Input
  { inputD         :: Int,
    inputPoints    :: V.Vector Double,
    inputCentroids :: V.Vector Double
  }

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o -> do
    points <- NE.map V.fromArray <$> o .: "points"
    centroids <- NE.map V.fromArray <$> o .: "centroids"
    pure $
      Input
        { inputD = V.length $ NE.head points,
          inputPoints = V.concat $ NE.toList points,
          inputCentroids = V.concat $ NE.toList centroids
        }

type CostOutput = Double

data DirOutput = DirOutput Int (V.Vector Double)

instance JSON.ToJSON DirOutput where
  toJSON (DirOutput d v) = JSON.toJSON $ getPoints d v

instance NFData DirOutput where
  rnf (DirOutput _ v) = rnf v

square :: (Num a) => a -> a
square x = x * x

dist2 :: (Num a) => [a] -> [a] -> a
dist2 a b = sum $ map square $ zipWith (-) a b

sum' :: (Num a) => [a] -> a
sum' = L.foldl' (+) 0

minimum' :: (Fractional a, Ord a) => [a] -> a
minimum' = L.foldl' min (1 / 0)

costGeneric :: (Fractional a, Ord a) => Int -> V.Vector a -> V.Vector a -> a
costGeneric d points centroids =
  sum' $
    map (\p -> minimum' $ map (dist2 p) centroids') $
      getPoints d points
  where
    centroids' = getPoints d centroids

cost :: Input -> CostOutput
cost (Input d points centroids) = costGeneric d points centroids

dir :: Input -> DirOutput
dir (Input d points centroids) =
  let (cost', cost'') =
        V.unzip $
          D.hessianProduct'
            (costGeneric d (fmap D.auto points))
            (V.map (,1) centroids)
   in DirOutput d $ V.zipWith (/) cost' cost''
