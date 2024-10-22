import "lib/github.com/diku-dk/linalg/linalg"

module linalg_f64 = mk_linalg f64

def fst (x,_) = x

def snd (_,y) = y

def sumBy 'a (f : a -> f64)  (xs : []a) : f64 = map f xs |> f64.sum

def l2normSq (v : []f64) = map (** 2) v |> f64.sum

def logsumexp = sumBy (f64.exp) >-> f64.log

def vMinus [m] (xs : [m]f64) (ys : [m]f64) : [m]f64 = zip xs ys |> map (\(x, y) -> x - y)

def frobeniusNormSq (mat : [][]f64) = flatten mat |> map (**2) |> f64.sum

def unpackQ [d] (logdiag: [d]f64) (lt: []f64) : [d][d]f64  =
  tabulate_2d d d (\i j ->
		    if i < j then 0
		    else if i == j then f64.exp logdiag[i]
		    else lt[d * j + i - j - 1 - j * (j + 1) / 2])

def logGammaDistrib (a : f64) (p : i64) =
  0.25 * f64.i64 p * f64.i64 (p - 1) * f64.log f64.pi +
  ((1...p) |> sumBy (\j -> f64.lgamma (a + 0.5 * f64.i64 (1 - j))))

def logsumexp_DArray (arr : []f64) =
    let mx = f64.maximum arr
    let sumShiftedExp = arr |> sumBy (\x -> f64.exp (x - mx))
    in f64.log sumShiftedExp + mx

def logWishartPrior [k] (qsAndSums: [k]([][]f64, f64)) wishartGamma wishartM p =
    let n = p + wishartM + 1
    let c = f64.i64 (n * p) * (f64.log wishartGamma - 0.5 * f64.log 2) - (logGammaDistrib (0.5 * f64.i64 n) p)
    let frobenius = qsAndSums |> sumBy (fst >-> frobeniusNormSq)
    let sumQs = qsAndSums |> sumBy snd
    in 0.5 * wishartGamma * wishartGamma * frobenius - f64.i64 wishartM * sumQs - f64.i64 k * c

def gmmObjective [d][k][n] (alphas: [k]f64) (means: [k][d]f64) (icf: [k][]f64) (x: [n][d]f64) (wishartGamma: f64) (wishartM: i64) =
    let constant = -(f64.i64 n * f64.i64 d * 0.5 * f64.log (2 * f64.pi))
    let alphasAndMeans = zip alphas means
    let qsAndSums = icf |> map (\v ->
                                let logdiag = v[0:d]
                                let lt = v[d:]
                                in (unpackQ logdiag lt, f64.sum logdiag))
    let slse = x |> sumBy (\xi ->
                    logsumexp_DArray <| map2
                        (\qAndSum alphaAndMeans ->
                            let (q, sumQ) = qAndSum
                            let (alpha, meansk) = alphaAndMeans
			    let qximeansk = flatten (transpose (linalg_f64.matmul q (transpose [vMinus xi meansk])))
                            in -0.5 * l2normSq qximeansk + alpha + sumQ
                        ) qsAndSums alphasAndMeans)
    in constant + slse  - f64.i64 n * logsumexp alphas + logWishartPrior qsAndSums wishartGamma wishartM d

def grad f x = vjp f x 1f64

entry calculate_objective [d][k]
			  (alphas: [k]f64)
			  (means: [k][d]f64)
			  (icf: [k][]f64)
			  (x: [][d]f64)
			  (w_gamma: f64) (w_m: i64) =
    gmmObjective alphas means icf x w_gamma w_m

entry calculate_jacobian [d][k]
			 (alphas: [k]f64)
			 (means: [k][d]f64)
			 (icf: [k][]f64)
			 (x: [][d]f64)
			 (w_gamma: f64) (w_m: i64) =
  grad (\(a, m, i) -> gmmObjective a m i x w_gamma w_m) (alphas, means, icf)
