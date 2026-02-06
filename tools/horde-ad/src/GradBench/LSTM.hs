{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module GradBench.LSTM (objective, jacobian) where

import Control.DeepSeq (NFData (..))
import Data.Aeson (ToJSON (..), (.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.List.NonEmpty qualified as NonEmpty
import Data.Vector.Storable qualified as VS
import HordeAd

type LSTMParams target =
  ( target (TKR 2 Double)  -- main_params  :: [stlen * 2][4 * d]
  , target (TKR 2 Double)  -- extra_params :: [3][d]
  )

data LSTMInputAux = LSTMInputAux
  { lstmStLen    :: Int
  , lstmD        :: Int
  , lstmLenSeq   :: Int
  , lstmState    :: Concrete (TKR 2 Double)  -- :: [stlen * 2][d]
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
    let lstmStLen = length main_params `div` 2
        lstmD = VS.length (sequenceArr !! 0)
        lstmLenSeq = length sequenceArr
        lstmMainParams = chunk (lstmStLen * 2) (4 * lstmD) main_params
        lstmExtraParams = chunk 3 lstmD extra_params
        lstmState = chunk (lstmStLen * 2) lstmD state
        lstmSequence = chunk lstmLenSeq lstmD sequenceArr
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

sigmoid :: ADReady target
        => target (TKR n Double) -> target (TKR n Double)
sigmoid x = let one = rrepl (rshape x) 1
            in one / (one + exp (- x))

lstmModel :: ADReady target
          => target (TKR 2 Double)  -- :: [4][d]
          -> target (TKR 2 Double)  -- :: [4][d]
          -> target (TKR 1 Double)  -- :: [d]
          -> target (TKR 1 Double)  -- :: [d]
          -> target (TKR 1 Double)  -- :: [d]
          -> (target (TKR 1 Double), target (TKR 1 Double))  -- :: ([d], [d])
lstmModel weight bias hidden cell input =
  let forget = sigmoid $ input * weight ! [0] + bias ! [0]
      ingate = sigmoid $ hidden * weight ! [1] + bias ! [1]
      outgate = sigmoid $ input * weight ! [2] + bias ! [2]
      change = tanh $ hidden * weight ! [3] + bias ! [3]
      t1s = cell * forget
      t2s = ingate * change
      cell2 = t1s + t2s
      hidden2 = tanh cell2 * outgate
  in (hidden2, cell2)

lstmPredict :: forall target. ADReady target
            => target (TKR 4 Double)  -- :: [stlen][2][4][d]
            -> target (TKR 2 Double)  -- :: [3][d]
            -> target (TKR 3 Double)  -- :: [stlen][2][d]
            -> Concrete (TKR 1 Double)  -- :: [d]
            -> (target (TKR 1 Double), target (TKR 3 Double))
                 -- :: ([d], [stlen][2][d])
lstmPredict mainParams extraParams state input =
  let x0 = rconcrete (unConcrete input) * extraParams ! [0]
      loop :: [target (TKR 2 Double)] -> target (TKR 1 Double) -> Int
           -> ([target (TKR 2 Double)], target (TKR 1 Double))
      loop l x i | i >= rwidth mainParams = (l, x)
      loop l x i =
        let (h, c) = lstmModel (mainParams ! [kconcrete i, 0])
                               (mainParams ! [kconcrete i, 1])
                               (state ! [kconcrete i, 0])
                               (state ! [kconcrete i, 1])
                               x
        in loop (rfromList [h, c] : l) h (i + 1)
      (l', x') = loop [] x0 0
      v' = x' * extraParams ! [1] + extraParams ! [2]
  in (v', rfromList $ NonEmpty.fromList $ reverse l')

lstmObjective :: forall target. ADReady target
              => LSTMInputAux -> LSTMParams target -> target (TKScalar Double)
lstmObjective LSTMInputAux{..} (lstmMainParams, lstmExtraParams) =
  let mainParams :: target (TKR 4 Double)  -- [stlen][2][4][d]
      mainParams = rreshape [lstmStLen, 2, 4, lstmD] lstmMainParams
      state :: target (TKR 3 Double)  -- [stlen][2][d]
      state = rconcrete $ unConcrete $ rreshape [lstmStLen, 2, lstmD] lstmState
      -- To represent this as rfold, we'd need to keep mainParams
      -- and lstmExtraParams unchaged in the accumulator (in addition
      -- to the changing oldState and oldTotal) and reformulate the loop
      -- to only use the i-th element of lstmSequence.
      loop :: Concrete (TKR 1 Double) -> target (TKR 3 Double)
           -> target (TKScalar Double) -> Int
           -> target (TKScalar Double)
      loop _ _ oldTotal i | i >= lstmLenSeq - 1 = oldTotal
      loop inputi oldState oldTotal i =
        let (y_pred, newState) =
              lstmPredict mainParams lstmExtraParams oldState inputi
            tmp_sum = rsum0 $ exp y_pred
            tmp_log = - log (tmp_sum + 2)
            ynorm = y_pred + rreplicate0N (rshape y_pred) tmp_log
            inputi1 = lstmSequence ! [kconcrete (i + 1)]
            newTotal = oldTotal + rdot0 (rconcrete $ unConcrete inputi1) ynorm
        in loop inputi1 newState newTotal (i + 1)
      total = loop (lstmSequence ! [0]) state 0 0
      count = lstmD * (lstmLenSeq - 1)
  in - total / kconcrete (fromIntegral count)

objective :: LSTMInput Concrete -> Double
objective (LSTMInput lstmInputAux lSTMParams) =
  unConcrete $ lstmObjective lstmInputAux lSTMParams


jacobian :: LSTMInput Concrete -> LSTMOutput
jacobian (LSTMInput lstmParams lstmInputAux) =
  LSTMOutput $ cgrad (lstmObjective lstmParams) lstmInputAux
