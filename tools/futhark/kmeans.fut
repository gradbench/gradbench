def euclid_dist_2 [d] (pt1: [d]f64) (pt2: [d]f64) : f64 =
  f64.sum (map (\x -> x * x) (map2 (-) pt1 pt2))

entry cost [n] [k] [d] (points: [n][d]f64) (centroids: [k][d]f64) =
  points
  |> map (\p -> map (euclid_dist_2 p) centroids)
  |> map f64.minimum
  |> f64.sum

entry dir [k] [n] [d] (points: [n][d]f64) (centroids: [k][d]f64) =
  let (cost', cost'') =
    jvp2 (\x -> vjp (cost points) x 1)
         centroids
         (replicate k (replicate d 1))
  in map2 (map2 (/)) cost' cost''
