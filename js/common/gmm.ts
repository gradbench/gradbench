import type { Float, Int, Runs } from "./protocol.ts";

/** Independent variables for which the gradient must be computed. */
export interface Independent {
  /** Parametrization for weights. */
  alpha: Float[];

  /** Means. */
  mu: Float[][];

  /** Logarithms of diagonal part for constructing precision matrices. */
  q: Float[][];

  /** Strictly lower triangular part for constructing precision matrices. */
  l: Float[][];
}

/** The full input. */
export interface Input extends Runs, Independent {
  /** Dimension of the space. */
  d: Int;

  /** Number of means. */
  k: Int;

  /** Number of points. */
  n: Int;

  /** Data points. */
  x: Float[][];

  /** Additional degrees of freedom in Wishart prior. */
  m: Int;

  /** Precision in Wishart prior. */
  gamma: Float;
}
