module GradBench.Hello
  ( square,
    double,
    Input,
    SquareOutput,
    DoubleOutput,
  )
where

import HordeAd (grad)
import HordeAd.Core.Adaptor

type Input = Double

type SquareOutput = Double

type DoubleOutput = Double

squareGeneric :: (Num a) => a -> a
squareGeneric x = x * x

square :: Input -> SquareOutput
square = squareGeneric

double :: Input -> DoubleOutput
double = fromDValue . grad squareGeneric . fromValue
