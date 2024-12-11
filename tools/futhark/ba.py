import futhark_server
import futhark_utils
import numpy as np


def prepare(server, input):
    n = input["n"]
    m = input["m"]
    p = input["p"]
    one_cam = input["cam"]
    one_X = input["x"]
    one_w = input["w"]
    one_feat = input["feat"]

    cams = np.tile(one_cam, (n, 1))
    X = np.tile(one_X, (m, 1))
    w = np.tile(one_w, p)
    feats = np.tile(one_feat, (p, 1))

    camIdx = 0
    ptIdx = 0
    obs = []
    for i in range(p):
        obs.append((camIdx, ptIdx))
        camIdx = (camIdx + 1) % n
        ptIdx = (ptIdx + 1) % m

    server.put_value("cams", cams)
    server.put_value("X", X)
    server.put_value("w", w)
    server.put_value("obs", np.array(obs, dtype=np.int32))
    server.put_value("feats", feats)


def objective(server, input):
    runs = input["runs"]
    (r_err, w_err), times = futhark_utils.run(
        server,
        "calculate_objective",
        ("reproj_error", "w_err"),
        ("cams", "X", "w", "obs", "feats"),
        runs,
    )
    num_r = r_err.shape[0]
    num_w = w_err.shape[0]
    return (
        {
            "reproj_error": {"elements": r_err[0].tolist(), "repeated": num_r},
            "w_err": {"element": w_err[0], "repeated": num_w},
        },
        times,
    )


def jacobian(server, input):
    runs = input["runs"]
    (rows, cols, vals), times = futhark_utils.run(
        server,
        "calculate_jacobian",
        ("rows", "cols", "vals"),
        ("cams", "X", "w", "obs", "feats"),
        runs,
    )
    return (
        {
            "BASparseMat": {
                "rows": rows.shape[0] - 1,
                "columns": int(cols[-1] + 1),
            }
        },
        times,
    )
