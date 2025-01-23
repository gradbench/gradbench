def euclid_dist_2 [d] (pt1: [d]f64) (pt2: [d]f64) : f64 =
  f64.sum (map (\x -> x * x) (map2 (-) pt1 pt2))

def cost [n] [k] [d] (points: [n][d]f64) (centres: [k][d]f64) =
  points
  |> map (\p -> map (euclid_dist_2 p) centres)
  |> map f64.minimum
  |> f64.sum

def tolerance = 1.0 : f64

def max_iterations : i32 = 10

entry kmeans [n] [d]
             (k: i64)
             (points: [n][d]f64) =
  let cluster_centres = take k (reverse points)
  let i = 0
  let stop = false
  let (cluster_centres, _i, _stop) =
    loop (cluster_centres: [k][d]f64, i, stop)
    while i < max_iterations && !stop do
      let (cost', cost'') =
        jvp2 (\x -> vjp (cost points) x 1)
             cluster_centres
             (replicate k (replicate d 1))
      let x = map2 (map2 (/)) cost' cost''
      let new_centres = map2 (map2 (-)) cluster_centres x
      let stop =
        (map2 euclid_dist_2 new_centres cluster_centres |> f64.sum)
        < tolerance
      in (new_centres, i + 1, stop)
  in cluster_centres
