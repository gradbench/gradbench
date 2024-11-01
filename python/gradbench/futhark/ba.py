import futhark_server
import numpy as np


def prepare(server, params):
    input = params["input"]
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


def calculate_objectiveBA(server):
    server.cmd_call(
        "calculate_objective", "reproj_error", "w_err", "cams", "X", "w", "obs", "feats"
    )
    r_err = server.get_value("reproj_error")
    w_err = server.get_value("w_err")
    num_r = r_err.shape[0]
    num_w = w_err.shape[0]
    return {
        "reproj_error": {"elements": r_err[0].tolist(), "repeated": num_r},
        "w_err": {"element": w_err[0], "repeated": num_w},
    }

def calculate_jacobianBA(server):
    server.cmd_call(
        "calculate_jacobian", "rows", "cols", "vals", "cams", "X", "w", "obs", "feats"
    )
    return {"BASparseMat": {"rows": server.get_value("rows").shape[0]-1,
                            "columns": int(server.get_value("cols")[-1]+1)}
            }
