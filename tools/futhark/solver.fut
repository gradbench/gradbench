module solver = {
  def magnitude_squared = map (\x -> x * x) >-> f64.sum

  def magnitude = magnitude_squared >-> f64.sqrt

  def vminus = map2 (f64.-)

  def ktimesv k = map (f64.* k)

  def distance_squared u v = magnitude_squared (u `vminus` v)

  def distance u v = f64.sqrt (distance_squared u v)

  def multivariate_argmin [n]
                          (grad: ([n]f64 -> f64) -> [n]f64 -> [n]f64)
                          (f: [n]f64 -> f64)
                          (x: [n]f64) : [n]f64 =
    let g = grad f
    let continue s = s.5
    let step (x, fx, gx, eta, i, _go) =
      if magnitude gx <= 1e-5
      then (x, fx, gx, eta, i, false)
      else if i == 10
      then (x, fx, gx, 2 * eta, 0, true)
      else let x_prime = x `vminus` (eta `ktimesv` gx)
           in if distance x x_prime <= 1e-5
              then (x, fx, gx, eta, i, false)
              else let fx_prime = f x_prime
                   in if fx_prime < fx
                      then (x_prime, fx_prime, g x_prime, eta, i + 1, true)
                      else (x, fx, gx, eta / 2, 0, true)
    in (iterate_while continue step (x, f x, g x, 1e-5, 0, true)).0

  def multivariate_argmax grad f x = multivariate_argmin grad (\x -> -(f x)) x

  def multivariate_max grad f x = f (multivariate_argmax grad f x)
}
