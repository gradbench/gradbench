-- | This is a somewhat inefficient implementation that directly
-- implements the recursive specification, using lists of lists to
-- represent the matrix. Remarkably, because the algorithm is
-- fundamentally so inefficient (something like O(n!)), the workloads
-- are necessarily tiny, and hence this inefficient representation is
-- adequate.
module GradBench.Det
  ( Input,
    PrimalOutput,
    GradientOutput,
    primal,
    gradient,
  )
where

import Data.Aeson ((.:))
import Data.Aeson qualified as JSON
import Data.List qualified as L
import Numeric.AD.Double qualified as D

data Input = Input
  { _inputA :: [Double],
    _inputEll :: Int
  }

type PrimalOutput = Double

type GradientOutput = [Double]

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "A" <*> o .: "ell"

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk n xs =
  let (bef, aft) = splitAt n xs
   in bef : chunk n aft

picks :: [a] -> [[a]]
picks l = zipWith (<>) (L.inits l) (map tail $ init $ L.tails l)

parts :: [[a]] -> [[[a]]]
parts = L.transpose . map picks . tail

minors :: (Fractional a) => [[a]] -> [a]
minors = map det . parts

det :: (Fractional a) => [[a]] -> a
det [[x]] = x
det a = sum $ do
  (f, aij, mij) <- zip3 (cycle [1, -1]) (head a) $ minors a
  pure $ fromIntegral (f :: Int) * aij * mij

primal :: Input -> PrimalOutput
primal (Input a ell) = det $ chunk ell a

gradient :: Input -> GradientOutput
gradient (Input a ell) = D.grad (det . chunk ell) a
