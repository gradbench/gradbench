import "lib/github.com/diku-dk/linalg/linalg"

module linalg_f64 = mk_linalg f64

def matmul = linalg_f64.matmul
def matadd = map2 (map2 (f64.+))

def identity n = tabulate_2d n n (\i j -> f64.bool(i == j))

def angle_axis_to_rotation_matrix (angle_axis: [3]f64) : [3][3]f64 =
  let n = f64.sqrt (angle_axis[0]**2+angle_axis[1]**2+angle_axis[2]**2)
  in if n < 0.0001 then identity 3 else
     let x = angle_axis[0] / n
     let y = angle_axis[1] / n
     let z = angle_axis[2] / n
     let s = f64.sin n
     let c = f64.cos n
     in [[x * x + (1 - x * x) * c,
          x * y * (1 - c) - z * s,
          x * z * (1 - c) + y * s],
         [x * y * (1 - c) + z * s,
          y * y + (1 - y * y) * c,
          y * z * (1 - c) - x * s],
         [x * z * (1 - c) - y * s,
          z * y * (1 - c) + x * s,
          z * z + (1 - z * z) * c]]

def apply_global_transform [n][m] (pose_params: [n][3]f64) (positions: [3][m]f64) : [3][m]f64 =
  let R = angle_axis_to_rotation_matrix pose_params[0]
          |> map (map2 (*) pose_params[1])
  in (R `matmul` positions) `matadd`
     transpose (replicate m [pose_params[2,0], pose_params[2,1], pose_params[2,2]])

def relatives_to_absolutes [n] (relatives: [n][4][4]f64) (parents: [n]i32) : [n][4][4]f64 =
  -- Initial value does not matter.
  loop absolutes : *[n][4][4]f64 = replicate n (identity 4)
  for i < n do
  let relative = relatives[i]
  let parent = parents[i]
  in absolutes with [i] = if parent == -1
                         then relative
                         else copy (absolutes[parent] `matmul` relative)

def euler_angles_to_rotation_matrix (xzy: [3]f64) : [4][4]f64 =
  let tx = xzy[0]
  let ty = xzy[2]
  let tz = xzy[1]
  let costx = f64.cos(tx)
  let sintx = f64.sin(tx)
  let costy = f64.cos(ty)
  let sinty = f64.sin(ty)
  let costz = f64.cos(tz)
  let sintz = f64.sin(tz)
  in [[costy * costz,
       -costx * sintz + sintx * sinty * costz,
       sintx * sintz + costx * sinty * costz,
       0],
      [costy * sintz,
       costx * costz + sintx * sinty * sintz,
       -sintx * costz + costx * sinty * sintz,
       0],
      [-sinty,
       sintx * costy,
       costx * costy,
       0],
      [0,
       0,
       0,
       1]]

type~ hand_model [num_bones][M] =
  { parents: [num_bones]i32,
    base_relatives: [num_bones][4][4]f64,
    inverse_base_absolutes: [num_bones][4][4]f64,
    weights: [num_bones][M]f64,
    base_positions: [4][M]f64,
    triangles: [][3]i32,
    is_mirrored: bool
  }

def get_posed_relatives [num_bones][M] (model: hand_model [num_bones][M]) (pose_params: [][3]f64) =
  let offset = 3
  let f i =
    matmul model.base_relatives[i]
           (euler_angles_to_rotation_matrix pose_params[i+offset])
  in tabulate num_bones f

def get_skinned_vertex_positions [num_bones][M]
                                 (model: hand_model [num_bones][M])
                                 (pose_params: [][3]f64)
                                 (apply_global: bool) =
  let relatives = get_posed_relatives model pose_params
  let absolutes = relatives_to_absolutes relatives model.parents
  let transforms = map2 matmul absolutes model.inverse_base_absolutes
  let base_positions = model.base_positions
  let positions =
    loop pos = tabulate_2d 3 M (\_ _ -> 0)
    for i < num_bones
    do let transform = transforms[i]
       let weights = model.weights[i]
       in map2 (map2 (+))
            pos
            (transform[0:3] `matmul` base_positions
             |> map (map2 (*) weights))

  let positions = if model.is_mirrored
                  then positions with [0] = map f64.neg positions[0]
                  else positions

  in if apply_global
     then apply_global_transform pose_params positions
     else positions

def to_pose_params (theta: []f64) (num_bones: i64) : [][]f64 =
  let n = 3 + num_bones
  let num_fingers = 5
  let cols = 5 + num_fingers * 4
  in tabulate n (\i -> match i
                       case 0 -> take 3 theta[0:]
                       case 1 -> [1,1,1]
                       case 2 -> take 3 theta[3:]
                       case j ->
                         if j >= cols || j == 3 || j % 4 == 0 then [0,0,0]
                         else if j % 4 == 1 then [theta[j + 1], theta[j + 2], 0]
                         else [theta[j + 2], 0, 0])

def (+^) = map2 (f64.+)

-- Not sure if this should be a run-time parameter, but it is constant
-- for all datasets, and seems more like an algorithmic property (it's
-- an encoding of various spatial transformations).
def theta_count : i64 = 26

def objective [num_us][num_bones][N][M]
    (model: hand_model [num_bones][M])
    (correspondences: [N]i32)
    (points: [3][N]f64)
    (theta: [theta_count]f64)
    (us: [num_us]f64) : [N][3]f64 =
  let pose_params = to_pose_params theta num_bones
  let vertex_positions = get_skinned_vertex_positions model pose_params true
  in if length us == 0
     then -- "Simple" case
       map2 (\point correspondence ->
               map2 (-) point vertex_positions[:, correspondence])
            (transpose points) correspondences
     else -- "Complex" case
     let us = unflatten (sized (N*2) us)
     in map3 (\point correspondence u ->
             let verts = model.triangles[correspondence]
             let hand_point =
               map (*u[0]) (vertex_positions[:, verts[0]])
               +^ map (*u[1]) vertex_positions[:, verts[1]]
               +^ map (*(1 - u[0] - u[1])) (vertex_positions[:, verts[2]])
             in map2 (-) point hand_point)
          (transpose points) correspondences us

-- All parameters up to and including 'is_mirrored' constitute the
-- model.  Of the remaining three parameters, 'theta' is called 'p' in
-- the paper.
entry calculate_objective [num_bones][N][M]
    (parents: [num_bones]i32)
    (base_relatives: [num_bones][4][4]f64)
    (inverse_base_absolutes: [num_bones][4][4]f64)
    (weights: [num_bones][M]f64)
    (base_positions: [4][M]f64)
    (triangles: [][3]i32)
    (is_mirrored: bool)
    (correspondences: [N]i32) (points: [3][N]f64) (theta: [theta_count]f64) (us: []f64)
    : [N][3]f64 =
  let model : hand_model [num_bones][M] =
    { parents,
      base_relatives,
      inverse_base_absolutes,
      weights,
      base_positions,
      triangles,
      is_mirrored }
  in objective model correspondences points theta us

import "lib/github.com/diku-dk/autodiff/onehot"

-- The Jacobian is morally transposed, because that is what ADBench expects.
entry calculate_jacobian [num_bones][N][M][num_us]
  (parents: [num_bones]i32)
  (base_relatives: [num_bones][4][4]f64)
  (inverse_base_absolutes: [num_bones][4][4]f64)
  (weights: [num_bones][M]f64)
  (base_positions: [4][M]f64)
  (triangles: [][3]i32)
  (is_mirrored: bool)
  (correspondences: [N]i32) (points: [3][N]f64) (theta: [theta_count]f64) (us: [num_us]f64)
  : [][N*3]f64 =
  let model : hand_model [num_bones][M] =
    { parents,
      base_relatives,
      inverse_base_absolutes,
      weights,
      base_positions,
      triangles,
      is_mirrored }
  let us_derivs = if num_us == 0 then 0 else 2
  let oh : onehot.gen [theta_count*1+us_derivs]
                      ([theta_count]f64,[num_us/2][2]f64) =
    onehot.(pair (arr f64) (resize (cycle (arr f64))))
  let f (theta',us') =
    let us'' = sized num_us (flatten us')
    in jvp (uncurry (objective model correspondences points))
           (theta,us) (theta',us'')
  let J = map flatten (map f (onehots oh))

  in if num_us == 0
     then J
     else
       -- ADBench expects the packed 'us' derivatives to be in the
       -- first two columns.
       rotate (-2) J
