{-# LANGUAGE OverloadedStrings #-}
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
import Data.Vector.Storable qualified as VS
import HordeAd

data Input = Input
  { _inputA :: VS.Vector Double,
    _inputEll :: Int
  }

type PrimalOutput = Double

type GradientOutput = VS.Vector Double

instance JSON.FromJSON Input where
  parseJSON = JSON.withObject "input" $ \o ->
    Input <$> o .: "A" <*> o .: "ell"

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk n xs =
  let (bef, aft) = splitAt n xs
   in bef : chunk n aft

picks :: [a] -> [[a]]
picks l = zipWith (<>) (L.inits l) (tail $ L.tails l)

parts :: [[a]] -> [[[a]]]
parts = L.transpose . map picks . tail

minors :: (Num a) => [[a]] -> [a]
minors = map det . parts

det :: (Num a) => [[a]] -> a
det [[x]] = x
det a = sum $ do
  (f, aij, mij) <- zip3 (cycle [1, -1]) (head a) $ minors a
  pure $ fromIntegral (f :: Int) * aij * mij

primal :: Input -> PrimalOutput
primal (Input a ell) = det $ chunk ell $ VS.toList a

gradient :: Input -> GradientOutput
gradient (Input a ell) =
  VS.fromList $ map unConcrete
  $ cgrad (det . chunk ell) (map kconcrete $ VS.toList a)
    -- Symbolic grad takes forever due to the build-up of product terms, because
    -- lists are represented as nested products, and due to lack of explicit
    -- sharing in the naively ported code. The former reason also makes
    -- the non-symbolic cgrad >15 times slower than the haskell-ad
    -- implementation, which we reuse here with minimal changes.
    -- This variant also takes a lot of memory to allocate the products on tape
    -- and so can't cope with size 11.
