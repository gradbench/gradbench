{-# LANGUAGE OverloadedLists, OverloadedStrings #-}
module GradBench.LSE
  ( primal,
    gradient,
  )
where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.Array.Nested qualified as Nested
import Data.Vector.Storable qualified as VS
import GHC.TypeLits (KnownNat)
import HordeAd hiding (rlogsumexp)
import HordeAd.Core.AstEnv
import HordeAd.Core.AstInterpret

newtype Input = Input (VS.Vector Double)

type PrimalOutput = Double

type GradientOutput = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> (o .: "x")

-- Copied from HordeAd.External.CommonRankedOps.
-- Fails for empty x'.
rlogsumexp :: (KnownNat n, NumScalar r, Differentiable r, ADReady target)
          => target (TKR n r) -> target (TKScalar r)
rlogsumexp x' = tlet x' $ \x -> tlet (rmaximum x) $ \maxx ->
  let shiftedx = x - rreplicate0N (rshape x) maxx
      logged = log (rsum0 (exp shiftedx))
  in logged + maxx

primal :: Input -> PrimalOutput
primal (Input x) =
  let y = rconcrete . Nested.rfromVector [VS.length x] $ x
      ast = simplifyInlineContract $ rlogsumexp @1 y
  in -- unsafePerformIO (threadDelay 1000000) `seq` traceShow ("primal", printAstPrettyButNested (simplifyInlineContract $ rlogsumexp @1 (AstVar @FullSpan (mkAstVarName (FTKR [100] (FTKScalar @Double)) (intToAstVarId 1))))) $
     unConcrete $ interpretAstFull emptyEnv ast

gradient :: Input -> GradientOutput
gradient (Input x) = Nested.rtoVector . unConcrete
                     . grad (rlogsumexp @1)
                     . rconcrete . Nested.rfromVector [VS.length x] $ x
  -- cgrad and the symbolic grad are here equally fast
