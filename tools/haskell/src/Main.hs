{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Control.Applicative
import Control.Exception (catch)
import Control.Monad (forever, guard)
import Data.Aeson (ToJSON (..), (.:))
import Data.Aeson qualified as JSON
import Data.ByteString.Char8 qualified as BS
import Data.List qualified as L
import Data.Maybe (fromMaybe, isJust)
import Data.Text qualified as T
import GradBench.Hello qualified
import System.Exit
import System.IO (hFlush, stdout)
import System.IO.Error (isEOFError)
import Prelude hiding (mod)

wrap ::
  (JSON.FromJSON a, JSON.ToJSON b) =>
  (a -> b) ->
  JSON.Value ->
  Either T.Text JSON.Value
wrap f input =
  case JSON.fromJSON input of
    JSON.Error e -> Left $ "Invalid input:\n" <> T.pack e
    JSON.Success v ->
      Right $ JSON.toJSON $ f v

modules :: [((T.Text, T.Text), JSON.Value -> Either T.Text JSON.Value)]
modules =
  [ (("hello", "square"), wrap GradBench.Hello.square),
    (("hello", "double"), wrap GradBench.Hello.double)
  ]
  where

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
main = forever loop `catch` onError
  where
    loop = do
      msg <-
        fromMaybe (error "line is not a JSON value") . JSON.decodeStrict
          <$> BS.getLine
      let reply i vs = do
            BS.putStrLn . BS.toStrict $ JSON.encode $ JSON.object $ ("id", toJSON i) : vs
            hFlush stdout
      case msg of
        MsgStart i _ -> reply i [("tool", "haskell")]
        MsgDefine i mod -> reply i [("success", toJSON $ knownModule mod)]
        MsgEvaluate i mod fun input ->
          case L.lookup (mod, fun) modules of
            Nothing ->
              reply
                i
                [ ("success", toJSON False),
                  ("error", toJSON $ "unknown function: " <> fun)
                ]
            Just f -> case f input of
              Left err ->
                reply
                  i
                  [ ("success", toJSON False),
                    ("error", toJSON err)
                  ]
              Right v ->
                reply
                  i
                  [ ("success", toJSON True),
                    ("output", v)
                  ]
        MsgUnknown i -> reply i []

    onError e =
      if isEOFError e
        then exitWith ExitSuccess
        else do
          print e
          exitWith $ ExitFailure 1

    knownModule mod = isJust $ L.find ((== mod) . fst . fst) modules
