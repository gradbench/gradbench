# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Incorporates code from: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteBA.jl

module BA

import Zygote
import GradBench

# FIXME: it is very expensive to redo all the input parsing here for
# every run. We absolutely must hoist it out into a "prepare" stage.
function objective(j)
    input = GradBench.BA.input_from_json(j)
    (r_err, w_err) =
        GradBench.BA.objective(input.cams,
                               input.X,
                               input.w,
                               input.obs,
                               input.feats)
    num_r = size(r_err, 2)
    num_w = size(w_err, 1)
    Dict("reproj_error" => Dict("elements" => r_err[:,1], "repeated" => num_r),
         "w_err" => Dict("element" => w_err[1], "repeated" => num_w)
         )
end

function pack(cam, X, w)
    [cam[:]; X[:]; w]
end

function unpack(packed)
    packed[1:end-4], packed[end-3:end-1], packed[end]
end

function compute_w_err(w)
    1.0 - w * w
end

compute_w_err_d = x -> Zygote.gradient(compute_w_err, x)[1]

function compute_reproj_err_d(params, feat)
    cam, X, w = unpack(params)
    GradBench.BA.compute_reproj_err(cam, X, w, feat)
end

function compute_ba_J(cams, X, w, obs, feats)
    n = size(cams, 2)
    m = size(X, 2)
    p = size(obs, 2)
    jacobian = GradBench.BA.SparseMatrix(n, m, p)
    reproj_err_d = zeros(2 * p, GradBench.BA.N_CAM_PARAMS + 3 + 1)
    for i in 1:p
        compute_reproj_err_d_i = x -> compute_reproj_err_d(x, feats[:, i])
        camIdx =  obs[1, i]
        ptIdx = obs[2, i]
        y, back = Zygote._pullback(compute_reproj_err_d_i, pack(cams[:, camIdx], X[:, ptIdx], w[i]))
        ylen = size(y, 1)
        J = hcat([ back(1:ylen .== j)[2] for j ∈ 1:ylen ]...)
        GradBench.BA.insert_reproj_err_block!(jacobian, i, camIdx, ptIdx, J')
    end
    for i in 1:p
        w_err_d_i = compute_w_err_d(w[i])
        GradBench.BA.insert_w_err_block!(jacobian, i, w_err_d_i)
    end
    jacobian
end

function jacobian(j)
    input = GradBench.BA.input_from_json(j)
    J = compute_ba_J(input.cams, input.X, input.w, input.obs, input.feats)
    GradBench.BA.dedup_jacobian(J)
end

GradBench.register!("ba", Dict(
    "objective" => objective,
    "jacobian" => jacobian
))

end
