def t (i: i64) (n: i64) = -1 + f64.i64 i * 2 / (f64.i64 n - 1)

entry primal [m] (x: [m]f64) (n: i64) =
  let f i =
    let ti = t i n
    let muls = scan (*) 1 (replicate m ti with [0] = 1)
    let g mul xj = -(xj * mul)
    in (f64.sgn ti + f64.sum (map2 g muls x)) ** 2
  in f64.sum (tabulate n f) / 2

entry gradient [m] (x: [m]f64) (n: i64) =
  vjp (\x' -> primal x' n) x 1
