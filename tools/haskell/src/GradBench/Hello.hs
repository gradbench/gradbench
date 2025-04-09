module GradBench.Hello
  ( square,
    double,
    Input,
    SquareOutput,
    DoubleOutput,
  )
where

import Data.Tuple (Solo (..))
import Numeric.AD.Double (grad)

type Input = Double

type SquareOutput = Double

type DoubleOutput = Double

squareGeneric :: (Num a) => a -> a
squareGeneric x = x * x

square :: Input -> SquareOutput
square x = x * x

double :: Input -> DoubleOutput
double x =
  let MkSolo y = grad (\(MkSolo x') -> squareGeneric x') (MkSolo x)
   in y
