def euclid_dist_2 [d] (pt1: [d]f64) (pt2: [d]f64) : f64 =
  f64.sum (map (\x -> x * x) (map2 (-) pt1 pt2))

def costfun [n] [k] [d] (points: [n][d]f64) (centres: [k][d]f64) =
  points
  |> map (\p -> map (euclid_dist_2 p) centres)
  |> map f64.minimum
  |> f64.sum

entry cost [n] [d]
           (k: i64)
           (points: [n][d]f64) =
  let cluster_centres = take k (reverse points)
  in costfun points cluster_centres

entry direction [n] [d]
                (k: i64)
                (points: [n][d]f64) =
  let cluster_centres = take k (reverse points)
  let (costfun', costfun'') =
    jvp2 (\x -> vjp (costfun points) x 1)
         cluster_centres
         (replicate k (replicate d 1))
  in map2 (map2 (/)) costfun' costfun''
