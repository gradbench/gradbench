def logsumexp [n] (x: [n]f64) : f64 =
  let A = f64.maximum x
  in map (\x' -> x' - A) x |> map f64.exp |> f64.sum |> f64.log |> (+ A)

entry primal = logsumexp

entry gradient x = vjp logsumexp x 1
