-- TODO
-- See https://github.com/tomsmeding/ADBench/blob/accelerate/tools/Accelerate/src/BA.hs and https://tomsmeding.com/f/master.pdf and https://github.com/gradbench/gradbench/tree/main/evals/ba

{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module GradBench.BA (objective, jacobian) where

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

type Pt3D = (Double, Double, Double)
type Pt2D = (Double, Double)

data Camera =
  Camera { camR     :: Pt3D
         , camC     :: Pt3D
         , camF     :: Double
         , camX0    :: Pt2D
         , camKappa :: Pt2D
         }

data Observation =
  Observation { obsCamIdx :: !Int
              , obsPtIdx  :: !Int
              }

data BAIn =
  BAIn { baInCams  :: Vector Camera       -- [n]
       , baInPts   :: Vector Pt3D         -- [m]
       , baInW     :: Vector Double       -- [p]
       , baInFeats :: Vector Pt2D         -- [p]
       , baInObs   :: Vector Observation  -- [p]
       }

data BAFVal =
  BAFVal { baFValReprojErr :: Vector Pt2D    -- [p]
         , baFValWErr      :: Vector Double  -- [p]
         }

data BAOut =
  BAOut { baOutRows :: Vector Int     -- [2p + p]
        , baOutCols :: Vector Int     -- [11n + 3m + p]
        , baOutVals :: Vector Double  -- [31p]
        }
