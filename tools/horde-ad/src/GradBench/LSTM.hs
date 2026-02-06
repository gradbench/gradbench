{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
module GradBench.LSTM (objective, jacobian) where

import Control.DeepSeq (NFData (..))
import Data.Aeson (ToJSON (..), (.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Proxy (Proxy (Proxy))
import Data.Vector.Storable qualified as VS
import HordeAd
import HordeAd.Core.AstEnv
import HordeAd.Core.AstInterpret

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
  in (outgate, cell2)

lstmPredict :: forall target. ADReady target
            => target (TKR 4 Double)  -- :: [stlen][2][4][d]
            -> target (TKR 2 Double)  -- :: [3][d]
            -> target (TKR 3 Double)  -- :: [stlen][2][d]
            -> target (TKR 1 Double)  -- :: [d]
            -> target (TKProduct (TKR 1 Double) (TKR 3 Double))
                 -- :: ([d], [stlen][2][d])
lstmPredict mainParams extraParams state input =
  let x0 = input * extraParams ! [0]
      loop :: forall f. ADReady f
           => f (TKR 1 Double)
           -> f (TKProduct (TKR 3 Double) (TKR 2 Double))
           -> f (TKProduct (TKR 1 Double) (TKR 2 Double))
      loop !x !el =
        let !(!o, !c') = lstmModel (tproject1 el ! [0])
                                   (tproject1 el ! [1])
                                   (tproject2 el ! [0])
                                   (tproject2 el ! [1])
                                   x
       in tlet c' $ \c ->
          tlet (tanh c * o) $ \ !h ->
            tpair h (rfromList [h, c])
  in withSNat (rwidth mainParams) $ \snat ->
    tmapAccumL Proxy snat
               (FTKR [rwidth input] FTKScalar)
               (FTKR [2, rwidth input] FTKScalar)
               (FTKProduct (FTKR [2, 4, rwidth input] FTKScalar)
                           (FTKR [2, rwidth input] FTKScalar))
               loop x0 (tpair mainParams state)

lstmObjective :: forall target. ADReady target
              => LSTMInputAux -> LSTMParams target -> target (TKScalar Double)
lstmObjective LSTMInputAux{..} (lstmMainParams, lstmExtraParams) =
  let mainParams :: target (TKR 4 Double)  -- [stlen][2][4][d]
      mainParams = rreshape [lstmStLen, 2, 4, lstmD] lstmMainParams
      state :: target (TKR 3 Double)  -- [stlen][2][d]
      state = rconcrete $ unConcrete $ rreshape [lstmStLen, 2, lstmD] lstmState
      loop :: ( target (TKR 1 Double)
              , target (TKR 3 Double)
              , target (TKScalar Double) )
           -> Concrete (TKR 1 Double)
           -> ( target (TKR 1 Double)
              , target (TKR 3 Double)
              , target (TKScalar Double) )
      loop !(!oldYnorm, !oldState, !oldTotal) input' =
        let input = rconcrete $ unConcrete input'
            prediction = lstmPredict mainParams lstmExtraParams oldState input
            y_pred = tproject1 prediction * lstmExtraParams ! [1]
                     + lstmExtraParams ! [2]
            newState = tproject2 prediction
            newTotal = oldTotal + rdot0 input oldYnorm
            tmp_sum = rsum0 $ exp y_pred
            tmp_log = - log (tmp_sum + 2)
            ynorm = y_pred + rreplicate0N (rshape y_pred) tmp_log
        in (ynorm, newState, newTotal)
      (_, _, total) = foldl' loop (rreplicate0N [lstmD] 0, state, 0)
                             (runravelToList lstmSequence)
      count = lstmD * (lstmLenSeq - 1)
  in - total / kconcrete (fromIntegral count)

objective :: LSTMInput Concrete -> Double
objective (LSTMInput lstmInputAux lSTMParams) =
  unConcrete $ lstmObjective lstmInputAux lSTMParams

{- This shares better, so it's for the symbolic pipeline, which is slower:
lstmObjective :: forall target. ADReady target
              => LSTMInputAux -> LSTMParams target -> target (TKScalar Double)
lstmObjective LSTMInputAux{..} (lstmMainParams, lstmExtraParams) =
  let mainParams :: target (TKR 4 Double)  -- [stlen][2][4][d]
      mainParams = rreshape [lstmStLen, 2, 4, lstmD] lstmMainParams
      state :: target (TKR 3 Double)  -- [stlen][2][d]
      state = rconcrete $ unConcrete $ rreshape [lstmStLen, 2, lstmD] lstmState
      -- To represent this as rfold, we need to keep mainParams
      -- and lstmExtraParams unchaged in the accumulator, in addition
      -- to the changing oldYnorm, oldState and oldTotal.
      loop :: forall f. ADReady f
           => f (TKProduct (TKProduct (TKR 4 Double) (TKR 2 Double))
                           (TKProduct (TKProduct (TKR 1 Double) (TKR 3 Double))
                                      (TKScalar Double)))
           -> f (TKR 1 Double)
           -> f (TKProduct (TKProduct (TKR 4 Double) (TKR 2 Double))
                           (TKProduct (TKProduct (TKR 1 Double) (TKR 3 Double))
                                      (TKScalar Double)))
      loop !old !input =
        let params = tproject1 $ tproject1 old
            eparams = tproject2 $ tproject1 old
            oldYnorm = tproject1 $ tproject1 $ tproject2 old
            oldState = tproject2 $ tproject1 $ tproject2 old
            oldTotal = tproject2 $ tproject2 old
            newTotal = oldTotal + rdot0 input oldYnorm
        in tlet (lstmPredict params eparams oldState input) $ \prediction ->
           tlet (tproject1 prediction * eparams ! [1]
                 + eparams ! [2]) $ \y_pred ->
             let newState = tproject2 prediction
                 tmp_sum = rsum0 $ exp y_pred
                 tmp_log = - log (tmp_sum + 2)
                 ynorm = y_pred + rreplicate0N (rshape y_pred) tmp_log
             in tpair (tproject1 old) (tpair (tpair ynorm newState) newTotal)
      res = withSNat lstmLenSeq $ \snat ->
        tfold snat
              (STKProduct (STKProduct (STKR (SNat @4) STKScalar)
                                      (STKR (SNat @2) STKScalar))
                          (STKProduct (STKProduct (STKR (SNat @1) STKScalar)
                                                  (STKR (SNat @3) STKScalar))
                                      STKScalar))
              (STKR (SNat @1) STKScalar)
              loop
              (tpair (tpair mainParams lstmExtraParams)
                     (tpair (tpair (rreplicate0N [lstmD] 0) state)
                            0))
              (rconcrete $ unConcrete lstmSequence)
      finalTotal = tproject2 $ tproject2 res
      count = lstmD * (lstmLenSeq - 1)
  in - finalTotal / kconcrete (fromIntegral count)

objective :: LSTMInput Concrete -> Double
objective (LSTMInput lstmInputAux (lstmMainParams, lstmExtraParams)) =
  let ast = simplifyInlineContract
            $ lstmObjective lstmInputAux
                            ( rconcrete $ unConcrete lstmMainParams
                            , rconcrete $ unConcrete lstmExtraParams )
  in -- traceShow ("primal", printAstPrettyButNested ast) $
     unConcrete $ interpretAstFull emptyEnv ast
-}

jacobian :: LSTMInput Concrete -> LSTMOutput
jacobian (LSTMInput lstmParams lstmInputAux) =
  LSTMOutput $ cgrad (lstmObjective lstmParams) lstmInputAux
