# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Incorporates code from: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteHT.jl

module HT

import Zygote
import GradBench

function interleave_rows(A, B, C)
    stacked = cat(A, B, C; dims=3)                 # Stack into a 3D array (rows × cols × 3)
    permuted = permutedims(stacked, (3, 1, 2))     # Now shape is (3 × rows × cols)
    reshaped = reshape(permuted, :, size(A, 2))    # Interleave rows by reshaping
    return reshaped
end

struct JacobianHT <: GradBench.HT.AbstractHT end
function (::JacobianHT)(input)
    complicated = !isempty(input.us)

    if complicated
        # Jacobian w.r.t. theta
        wrapper_theta = θ -> GradBench.HT.objective_complicated(
            input.model, input.correspondences, input.points, θ, input.us)
        _, J_theta = Zygote.forward_jacobian(wrapper_theta, input.theta)

        # Jacobian w.r.t. each u in us
        n_us = size(input.us, 1)

        wrapper_u = us -> GradBench.HT.objective_complicated(
            input.model,
            input.correspondences,
            input.points,
            input.theta,
            us)
        _, back = Zygote.pullback(wrapper_u, input.us)

        seed = zeros(n_us*3)
        seed[1:3:end] .= 1
        x = back(reshape(seed, n_us, 3))[1]

        seed = zeros(n_us*3)
        seed[2:3:end] .= 1
        y = back(reshape(seed, n_us, 3))[1]

        seed = zeros(n_us*3)
        seed[3:3:end] .= 1
        z = back(reshape(seed, n_us, 3))[1]

        us_theta = interleave_rows(x,y,z)

        return vcat(us_theta', J_theta)
    else
        # Simple case: only theta matters
        _, J_theta = Zygote.forward_jacobian(θ -> GradBench.HT.objective_simple(
            input.model, input.correspondences, input.points, θ), input.theta)
        return J_theta
    end
end


GradBench.register!("ht", Dict(
    "objective" => GradBench.HT.ObjectiveHT(),
    "jacobian" => JacobianHT()
))

end
