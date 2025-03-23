-- Futhark does not support recursion, so instead we use a different
-- formulation of the determinant, found on
--
--   https://mathworld.wolfram.com/DeterminantExpansionbyMinors.html
--
-- This is not a computational advantage (except that it is data
-- parallel); indeed it is likely that the laborious computation of
-- permutations is significantly slower than the recursive
-- formulation. The amount of floating-point work is the same.

def fact (n: i64) = loop x = 1 for i < n do x * (i + 1)

-- Given the lexicographic index of a permutation, compute that
-- permutation.
def idx_to_perm (n: i64) (idx: i64) : [n]i64 =
  let perm = replicate n (-1)
  let elements = zip (iota n) (rep false)
  let fi = fact n
  in (.0)
     <| loop (perm, elements, idx, fi) for i in n - 1..n - 2...0 do
          let fi = fi / (i + 1)
          let pos = idx / fi
          let (elem, _) =
            loop (elem, pos) = (0, pos)
            while pos > 0 || elements[elem].1 do
              if elements[elem].1
              then (elem + 1, pos)
              else (elem + 1, pos - 1)
          let elements[elem] = elements[elem] with 1 = true
          in ( perm with [n - i - 1] = elements[elem].0
             , elements
             , idx % fi
             , fi
             )

-- Compute the inversion number from a lexicographic index of a
-- permutation.
def inversion_number_from_idx (n: i64) (idx: i64) =
  (.0)
  <| loop (sum, idx, fi) = (0, idx, fact n)
     for i in n - 1..n - 2..>0 do
       let fi = fi / (i + 1)
       in (sum + idx / fi, idx % fi, fi)

entry primal (ell: i64) (A: [ell * ell]f64) =
  let A = unflatten A
  in f64.sum (tabulate (fact ell)
                       (\pi ->
                          let p = idx_to_perm ell pi
                          in f64.i64 ((-1) ** inversion_number_from_idx ell pi)
                             * f64.product (tabulate ell (\i -> A[i][p[i]]))))

entry gradient (ell: i64) (A: [ell * ell]f64) =
  vjp (primal ell) A 1
