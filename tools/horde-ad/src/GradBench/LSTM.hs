{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module GradBench.LSTM (objective, jacobian) where

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

type LSTMParams target =
  ( target (TKR 2 Double)  -- main_params  :: [stlen * 2][4 * d]
  , target (TKR 2 Double)  -- extra_params :: [3][d]
  )

data LSTMInputAux = LSTMInputAux
  { lstmState    :: Concrete (TKR 2 Double)  -- :: [stlen * 2][d]
  , lstmSequence :: Concrete (TKR 2 Double)  -- :: [lenSeq][d]
  }

type role LSTMInput representational
data LSTMInput target = LSTMInput LSTMInputAux (LSTMParams target)

newtype LSTMOutput = LSTMOutput (LSTMParams Concrete)

instance JSON.FromJSON (LSTMInput Concrete) where
  parseJSON = JSON.withObject "input" $ \o -> do
    main_params <- o .: "main_params"
    extra_params <- o .: "extra_params"
    state <- o .: "state"
    sequenceArr <- o .: "sequence"
    let stlen = length main_params `div` 2
        d = VS.length (sequenceArr !! 0)
        lenseq = length sequenceArr
        lstmMainParams = chunk (stlen * 2) (4 * d) main_params
        lstmExtraParams = chunk 3 d extra_params
        lstmState = chunk (stlen * 2) d state
        lstmSequence = chunk lenseq d sequenceArr
        lstmInputAux = LSTMInputAux {..}
    pure $ LSTMInput lstmInputAux (lstmMainParams, lstmExtraParams)

instance JSON.ToJSON LSTMOutput where
  toJSON (LSTMOutput (lstmMainParams, lstmExtraParams)) =
    JSON.toJSON (unChunk lstmMainParams VS.++ unChunk lstmExtraParams)

chunk :: Int -> Int -> [VS.Vector Double] -> Concrete (TKR 2 Double)
chunk k d l =
  let xs = VS.concat l
  in Concrete $ Nested.rfromVector [k, d] xs

unChunk :: Concrete (TKR 2 Double) -> VS.Vector Double
unChunk = Nested.rtoVector . unConcrete

instance NFData LSTMOutput where
  rnf (LSTMOutput (!_, _)) = ()

lstmObjective :: ADReady target
              => LSTMInputAux -> LSTMParams target -> target (TKScalar Double)
lstmObjective LSTMInputAux{..} (lstmMainParams, lstmExtraParams) = 0

objective :: LSTMInput Concrete -> Double
objective (LSTMInput lstmInputAux (lstmMainParams, lstmExtraParams)) =
  let ast = simplifyInlineContract
            $ lstmObjective lstmInputAux
                            ( rconcrete $ unConcrete lstmMainParams
                            , rconcrete $ unConcrete lstmExtraParams )
  in -- traceShow ("primal", printAstPrettyButNested ast) $
     unConcrete $ interpretAstFull emptyEnv ast

jacobian :: LSTMInput Concrete -> LSTMOutput
jacobian (LSTMInput lstmParams lstmInputAux) =
  LSTMOutput $ grad (lstmObjective lstmParams) lstmInputAux
