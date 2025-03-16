import "solver"

def saddle grad0 grad1 (start: [2]f64) : [2][2]f64 =
  let f x1 y1 x2 y2 = (x1 ** 2 + y1 ** 2) - (x2 ** 2 + y2 ** 2)
  let r2_cost p1 p = f p1[0] p1[1] p[0] p[1]
  let r1_cost p1 = solver.multivariate_max grad1 (r2_cost p1) start
  let r1 = solver.multivariate_argmin grad0 r1_cost start
  let r2 = solver.multivariate_argmax grad0 (r2_cost r1) start
  in [r1, r2]

def grad_r f x = vjp f x 1f64
def grad_f f x = map (\i -> jvp f x (map (const 0) x with [i] = 1f64)) (indices x)

entry saddle_ff = saddle grad_f grad_f
entry saddle_fr = saddle grad_f grad_r
entry saddle_rr = saddle grad_r grad_r
entry saddle_rf = saddle grad_r grad_f
