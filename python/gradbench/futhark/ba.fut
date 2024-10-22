import "lib/github.com/athas/vector/vspace"

module v3d = mk_vspace_3d f64
type point_3d = v3d.vector
def point_3d a: point_3d = {x=a[0], y=a[1], z=a[2]}

module v2d = mk_vspace_2d f64
type point_2d = v2d.vector
def point_2d a: point_2d = {x=a[0], y=a[1]}

def rodrigues_rotate_point (rot: point_3d) (X: point_3d) =
    let sqtheta = v3d.quadrance rot in
    if sqtheta != 0 then
        let theta = f64.sqrt sqtheta
        let costheta = f64.cos theta
        let sintheta = f64.sin theta
        let theta_inv = 1 / theta

        let w = v3d.scale theta_inv rot
        let w_cross_X = v3d.cross w X
        let tmp = v3d.dot w X * (1 - costheta)

        in v3d.(scale costheta X +
                scale sintheta w_cross_X +
                scale tmp w)
    else v3d.(X + cross rot X)

def radial_distort (rad_params: point_2d) (proj: point_2d) =
  let rsq = v2d.quadrance proj
  let L = 1 + rad_params.x * rsq + rad_params.y * rsq * rsq
  in v2d.scale L proj

type cam = {rot: point_3d, center: point_3d, focal: f64, x0: point_2d, rad: point_2d}

def project (cam: cam) X =
  let Xcam = rodrigues_rotate_point cam.rot (X v3d.- cam.center)
  let distorted = radial_distort cam.rad (v2d.scale (1/Xcam.z) {x=Xcam.x, y=Xcam.y})
  in cam.x0 v2d.+ v2d.scale cam.focal distorted

def compute_reproj_err cam X w feat : point_2d =
  v2d.scale w (project cam X v2d.- feat)

def compute_zach_weight_error w : f64 =
  1 - w*w

def ba_objective [n][m][p] (cams: [n]cam) (X: [m]point_3d) (w: [p]f64) (obs:[p][2]i32) (feat:[]point_2d) =
  let reproj_err =
    tabulate p (\i -> compute_reproj_err cams[obs[i,0]]
                                         X[obs[i,1]]
                                         w[i]
                                         feat[i])
  let w_err = map compute_zach_weight_error w
  in (reproj_err, w_err)

def grad f x = vjp f x 1f64

def ba_diff [n][m][p] (cams: [n]cam)
                      (X: [m]point_3d)
                      (w: [p]f64)
                      (obs: [p][2]i32)
                      (feats: [p]point_2d) =
  let compute_reproj_err_J_block cam X w feat =
    let wrt_x (cam, X, w) = (compute_reproj_err cam X w feat).x
    let wrt_y (cam, X, w) = (compute_reproj_err cam X w feat).y
    let (dcam1, dX1, dw1) = grad wrt_x (cam, X, w)
    let (dcam2, dX2, dw2) = grad wrt_y (cam, X, w)
    in ((dcam1,dcam2), (dX1,dX2), (dw1, dw2))
  in map4 compute_reproj_err_J_block
          (map (\i -> cams[i]) obs[:,0])
          (map (\i -> X[i]) obs[:,1])
          w feats

def idx_cam (i: i64) (cam: cam) : f64 =
  match i
  case 0 -> cam.rot.x
  case 1 -> cam.rot.y
  case 2 -> cam.rot.z
  case 3 -> cam.center.x
  case 4 -> cam.center.y
  case 5 -> cam.center.z
  case 6 -> cam.focal
  case 7 -> cam.x0.x
  case 8 -> cam.x0.y
  case 9 -> cam.rad.x
  case _ -> cam.rad.y

def unpack_cam (cam: [11]f64) : cam =
  {rot = point_3d cam[0:3],
   center = point_3d cam[3:6],
   focal = cam[6],
   x0 = point_2d cam[7:9],
   rad = point_2d cam[9:11]}

entry calculate_objective [n][m][p] (cams: [n][11]f64) (X: [m][3]f64) (w: [p]f64) (obs:[p][2]i32) (feat:[][2]f64) =
  let X = map (\p -> {x=p[0],y=p[1],z=p[2]}) X
  let feat = map (\p -> {x=p[0],y=p[1]}) feat
  let (a,b) = ba_objective (map unpack_cam cams) X w obs feat
  in (map (\{x,y} -> [x,y]) a, b)

-- The packing code is derived from
-- https://github.com/tomsmeding/ADBench/blob/157260330293a46068593357cebc0f71f203750b/tools/Accelerate/src/BA.hs#L85-L138
-- by Tom Smeding.
entry calculate_jacobian [n][m][p] (cams: [n][11]f64) (X: [m][3]f64) (w: [p]f64) (obs:[p][2]i32) (feat:[p][2]f64) =
  let cams = map unpack_cam cams
  let X = map (\p -> {x=p[0],y=p[1],z=p[2]}) X
  let feat = map (\p -> {x=p[0],y=p[1]}) feat

  -- Compute derivatives.
  let grads = #[noinline] ba_diff cams X w obs feat

  -- Convert to matrix format.
  let dcams i j =
    let ((dcam1, dcam2), _, _) = grads[i/2]
    in idx_cam j (if (i % 2 == 0) then dcam1 else dcam2)

  let dX i j =
    let (_, (dX1, dX2), _) = grads[i/2]
    in if (i % 2 == 0)
       then if j == 0 then dX1.x
            else if j == 1 then dX1.y
            else dX1.z
       else if j == 0 then dX2.x
            else if j == 1 then dX2.y
            else dX2.z

  let drdw i =
    let (_, _, (dw1, dw2)) = grads[i/2]
    in if (i % 2 == 0) then dw1 else dw2

  let dwdw = map (grad compute_zach_weight_error) w

  -- Pack sparse Jacobian.
  let p32 = i32.i64 p
  let n32 = i32.i64 n
  let m32 = i32.i64 m
  let rows = map (*15) (0..<2*p32) ++ map (+(2*i32.i64 p*15)) (0..<p32+1)
  let colsvals =
    tabulate (2*p*15)
             (\i -> let obsi = i / (2*15)
                    let valouti = i / 15
                    let (ci, pi) = (i64.i32 obs[obsi,0], i64.i32 obs[obsi,1])
                    let j = i % 15
                    in if j < 11
                       then (i32.i64 (11*ci+j), dcams valouti j)
                       else if j < 14
                       then (11*n32+3*i32.i64 pi+i32.i64 j-11, dX valouti (j-11))
                       else (11*n32+3*m32+i32.i64 obsi, drdw valouti))
             ++ tabulate p (\i -> (11*n32+3*m32+i32.i64 i, dwdw[i]))

  in (rows, map (.0) colsvals, map (.1) colsvals)
