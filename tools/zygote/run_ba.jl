# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Incorporates code from: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteBA.jl

module BA

import Zygote
import GradBench

function pack(cam, X, w)
    return [cam[:]; X[:]; w]
end

function unpack(packed)
    return packed[1:(end - 4)], packed[(end - 3):(end - 1)], packed[end]
end

function compute_w_err(w)
    return 1.0 - w * w
end

compute_w_err_d = x -> Zygote.gradient(compute_w_err, x)[1]

function compute_reproj_err_d(params, feat)
    cam, X, w = unpack(params)
    return GradBench.BA.compute_reproj_err(cam, X, w, feat)
end

function compute_ba_J(cams, X, w, obs, feats)
    n = size(cams, 2)
    m = size(X, 2)
    p = size(obs, 2)
    jacobian = GradBench.BA.SparseMatrix(n, m, p)

    for i in 1:p
        camIdx = obs[1, i]
        ptIdx = obs[2, i]
        cam = cams[:, camIdx]
        pt = X[:, ptIdx]
        weight = w[i]
        feat = feats[:, i]

        compute_reproj_err_d_i(x) = compute_reproj_err_d(x, feat)
        packed_input = pack(cam, pt, weight)
        y, back = Zygote._pullback(compute_reproj_err_d_i, packed_input)

        # Compute full Jacobian row-wise using pullback
        J = map(j -> back(reshape(1.0 .* (1:length(y) .== j), size(y)))[2], 1:length(y))
        Jmat = hcat(J...)'

        GradBench.BA.insert_reproj_err_block!(jacobian, i, camIdx, ptIdx, Jmat)
    end

    for i in 1:p
        GradBench.BA.insert_w_err_block!(jacobian, i, compute_w_err_d(w[i]))
    end

    return jacobian
end

struct JacobianBA <: GradBench.BA.AbstractBA end
function (::JacobianBA)(input)
    J = compute_ba_J(input.cams, input.X, input.w, input.obs, input.feats)
    return GradBench.BA.dedup_jacobian(J)
end


GradBench.register!(
    "ba", Dict(
        "objective" => GradBench.BA.ObjectiveBA(),
        "jacobian" => JacobianBA()
    )
)

end
