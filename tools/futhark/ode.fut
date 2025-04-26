-- Pull-arrays.
type^ arr [n] 't = ([0][n](), i64 -> t)

def arr [n] f : arr [n] f64 =
  ([], f)

def idx [n] ((_, f): arr [n] f64) i =
  f i

def manifest [n] (f: arr [n] f64) =
  tabulate n (idx f)

def vecadd [n] (xs: arr [n] f64) (ys: arr [n] f64) : arr [n] f64 =
  arr (\i -> idx xs i + idx ys i)

def scale [n] x (v: arr [n] f64) : arr [n] f64 =
  arr (\i -> x * idx v i)

def runge_kutta [n] (f: arr [n] f64 -> arr [n] f64) (yi: [n]f64) (tf: f64) (s: i64) =
  let h = tf / f64.i64 s
  in loop yf = yi
     for _i < s do
       let yf = arr (\i -> yf[i])
       let k1 = f yf
       let k2 = f (yf `vecadd` scale (h / 2) k1)
       let k3 = f (yf `vecadd` scale (h / 2) k2)
       let k4 = f (yf `vecadd` scale h k3)
       in manifest (yf
                    `vecadd` (scale (h / 6)
                                    (k1
                                     `vecadd` scale 2 k2
                                     `vecadd` scale 2 k3
                                     `vecadd` k4)))

def ode_fun_vec [n] (x: [n]f64) (y: arr [n] f64) : arr [n] f64 =
  let f i = if i == 0 then x[i] else x[i] * idx y (i - 1 % n)
  in arr f

def tf : f64 = 2

entry primal [n] (x: [n]f64) (s: i64) : []f64 =
  runge_kutta (ode_fun_vec x) (map (const 0) x) tf s

entry gradient [n] (x: [n]f64) (s: i64) : []f64 =
  vjp (\x' -> last (primal x' s)) x 1
