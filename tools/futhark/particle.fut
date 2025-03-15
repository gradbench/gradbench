import "solver"

type p = (f64, f64)

def pplus (u: p) (v: p) = (u.0 + v.0, u.1 + v.1)

def ktimesp k (u: p) = (k * u.0, k * u.1)

def sqr (x: f64) = x * x

def pdistance (u: p) (v: p) = f64.sqrt ((sqr (u.0 - v.0) + sqr (u.1 - v.1)))

def naive_euler grad (w: f64) =
  let charges = [(10, 10 - w), (10, 0)]
  let x_initial = (0, 8)
  let xdot_initial = (0.75, 0)
  let delta_t = 1e-1
  let p x = f64.sum (map (f64.recip <-< pdistance x) charges)
  let (x, xdot, _) =
    loop (x, xdot, go) = (x_initial, xdot_initial, true)
    while go do
      let xddot = -1 `ktimesp` grad p x
      let x_new = x `pplus` (delta_t `ktimesp` xdot)
      in if x_new.1 > 0
         then (x_new, xdot `pplus` (delta_t `ktimesp` xddot), true)
         else (x, xdot, false)
  let delta_t_f = -x.1 / xdot.1
  let x_t_f = x `pplus` (delta_t_f `ktimesp` xdot)
  in sqr x_t_f.0

def particle grad0 grad1 w0 =
  solver.multivariate_argmin grad0 (\w -> naive_euler grad1 w[0]) [w0]

def grad_r f x = vjp f x 1f64
def grad_f f x = map (\i -> jvp f x (map (const 0) x with [i] = 1f64)) (indices x)
def pgrad_f f (x, y) = (jvp f (x, y) (1, 0), jvp f (x, y) (0f64, 1f64))

entry particle_ff = particle grad_f pgrad_f
entry particle_fr = particle grad_f grad_r
entry particle_rr = particle grad_r grad_r
entry particle_rf = particle grad_r pgrad_f
