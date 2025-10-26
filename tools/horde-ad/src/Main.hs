{-# LANGUAGE OverloadedStrings #-}
-- We limit the memory usage of evaluation with enableAllocationLimit,
-- as the tape-based code is otherwise very hungry. Note that this is
-- not a residency limit, but simply counts how many bytes are
-- allocated. On the other hand, we can reliably catch violations. It
-- is also appropriate for our use case where the vast majority of
-- allocations goes to the tape, which is kept until the very end.
-- Still, this precise size can be calibrated as necessary.
module Main (main) where

import Control.Applicative
import Control.DeepSeq (NFData, rnf)
import Control.Exception (SomeException, catch, evaluate, fromException, throw)
import Control.Monad (forever, guard)
import Data.Aeson (ToJSON (..), (.:))
import Data.Aeson qualified as JSON
import Data.ByteString.Char8 qualified as BS
import Data.List qualified as L
import Data.Maybe (fromMaybe, isJust)
import Data.Text qualified as T
import GradBench.Det qualified
import GradBench.Hello qualified
--import GradBench.KMeans qualified
--import GradBench.LLSq qualified
import GradBench.LSE qualified
import GradBench.ODE qualified
--import GradBench.Particle qualified
--import GradBench.Saddle qualified
import Prelude hiding (mod)
import System.Clock (Clock (Monotonic), getTime, toNanoSecs)
import System.Exit
import System.IO
import System.IO.Error (isEOFError)

data Runs = Runs
  { minRuns :: Int,
    minSeconds :: Double
  }

instance JSON.FromJSON Runs where
  parseJSON = JSON.withObject "input" $ \o ->
    Runs <$> o .: "min_runs" <*> o .: "min_seconds"

getRuns :: JSON.Value -> Runs
getRuns input =
  case JSON.fromJSON input of
    JSON.Success runs -> runs
    JSON.Error _ -> Runs 1 0

type Runtime = Integer

doRuns :: (NFData b) => [Runtime] -> Int -> Double -> Runs -> (a -> b) -> a -> IO (b, [Runtime])
doRuns timings i elapsed runs f x = do
  bef <- getTime Monotonic
  let output = f x
  evaluate $ rnf output
  aft <- getTime Monotonic
  let ns = toNanoSecs $ aft - bef
      timings' = ns : timings
      elapsed' = elapsed + fromIntegral ns / 1e9
  if i < minRuns runs || elapsed' < minSeconds runs
    then doRuns timings' (i + 1) elapsed' runs f x
    else pure (output, timings')

wrap ::
  (JSON.FromJSON a, JSON.ToJSON b, NFData b) =>
  (a -> b) ->
  JSON.Value ->
  IO (Either T.Text (JSON.Value, [Runtime]))
wrap f input =
  case JSON.fromJSON input of
    JSON.Error e ->
      pure $ Left $ "Invalid input:\n" <> T.pack e
    JSON.Success v -> do
      (output, timings) <- doRuns [] 1 0 (getRuns input) f v
      pure $ Right (JSON.toJSON output, timings)

modules ::
  [ ( (T.Text, T.Text),
      JSON.Value -> IO (Either T.Text (JSON.Value, [Runtime]))
    )
  ]
modules =
  [ (("hello", "square"), wrap GradBench.Hello.square),
    (("hello", "double"), wrap GradBench.Hello.double){-,
    (("kmeans", "cost"), wrap GradBench.KMeans.cost),
    (("kmeans", "dir"), wrap GradBench.KMeans.dir),
    (("llsq", "primal"), wrap GradBench.LLSq.primal),
    (("llsq", "gradient"), wrap GradBench.LLSq.gradient)-},
    (("ode", "primal"), wrap GradBench.ODE.primal),
    (("ode", "gradient"), wrap GradBench.ODE.gradient),
    (("det", "primal"), wrap GradBench.Det.primal),
    (("det", "gradient"), wrap GradBench.Det.gradient),
    (("lse", "primal"), wrap GradBench.LSE.primal),
    (("lse", "gradient"), wrap GradBench.LSE.gradient){-,
    (("particle", "rr"), wrap GradBench.Particle.rr),
    (("particle", "fr"), wrap GradBench.Particle.fr),
    (("particle", "rf"), wrap GradBench.Particle.rf),
    (("particle", "ff"), wrap GradBench.Particle.ff),
    (("saddle", "rr"), wrap GradBench.Saddle.rr),
    (("saddle", "fr"), wrap GradBench.Saddle.fr),
    (("saddle", "rf"), wrap GradBench.Saddle.rf),
    (("saddle", "ff"), wrap GradBench.Saddle.ff) -}
  ]

type Id = Int

-- | A message sent from the eval to us.
data Msg
  = MsgStart Id T.Text
  | MsgDefine Id T.Text
  | MsgEvaluate
      Id
      -- | Module
      T.Text
      -- | Function
      T.Text
      JSON.Value
  | MsgUnknown Id
  deriving (Show)

msgId :: Msg -> Id
msgId (MsgStart i _) = i
msgId (MsgDefine i _) = i
msgId (MsgEvaluate i _ _ _) = i
msgId (MsgUnknown i) = i

instance JSON.FromJSON Msg where
  parseJSON = JSON.withObject "Msg" $ \o -> do
    i <- o .: "id"
    pMsgStart i o
      <|> pMsgDefine i o
      <|> pMsgEvaluate i o
      <|> pure (MsgUnknown i)
    where
      pMsgStart i o = do
        guard . (== ("start" :: T.Text)) =<< (o .: "kind")
        MsgStart i <$> o .: "eval"
      pMsgDefine i o = do
        guard . (== ("define" :: T.Text)) =<< (o .: "kind")
        MsgDefine i <$> o .: "module"
      pMsgEvaluate i o = do
        guard . (== ("evaluate" :: T.Text)) =<< (o .: "kind")
        MsgEvaluate i <$> o .: "module" <*> o .: "function" <*> o .: "input"

main :: IO ()
main = forever loop
  where
    loop = do
      msg <-
        fromMaybe (error "line is not a JSON value") . JSON.decodeStrict
          <$> BS.getLine
            `catch` onReadError
      let reply vs = do
            BS.putStrLn . BS.toStrict . JSON.encode . JSON.object $
              ("id", toJSON (msgId msg)) : vs
            hFlush stdout
      case msg of
        MsgStart _ _ -> reply [("tool", "horde-ad")]
        MsgDefine _ mod -> reply [("success", toJSON $ knownModule mod)]
        MsgEvaluate _ mod fun input ->
          case L.lookup (mod, fun) modules of
            Nothing ->
              reply
                [ ("success", toJSON False),
                  ("error", toJSON $ "unknown function: " <> fun)
                ]
            Just f -> do
              r <- f input
              case r of
                Left err ->
                  reply
                    [ ("success", toJSON False),
                      ("error", toJSON err)
                    ]
                Right (v, runtimes) -> do
                  let mkTiming t =
                        JSON.object
                          [ ("name", "evaluate"),
                            ("nanoseconds", toJSON t)
                          ]
                  reply
                    [ ("success", toJSON True),
                      ("output", v),
                      ("timings", toJSON $ map mkTiming runtimes)
                    ]
        MsgUnknown _ -> reply []

    onReadError :: SomeException -> IO a
    onReadError e =
      if maybe False isEOFError $ fromException e
        then exitSuccess
        else throw e

    knownModule mod = isJust $ L.find ((== mod) . fst . fst) modules
