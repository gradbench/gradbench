# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Incorporates code from: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteHT.jl

module HT

import Zygote
import GradBench

# FIXME: it is very expensive to redo all the input parsing here for
# every run. We absolutely must hoist it out into a "prepare" stage.
function objective(j)
    input = GradBench.HT.input_from_json(j)
    complicated = size(input.us, 1) != 0
    if complicated
        GradBench.HT.objective_complicated(input.model,
                                           input.correspondences,
                                           input.points,
                                           input.theta,
                                           input.us)
    else
        GradBench.HT.objective_simple(input.model,
                                      input.correspondences,
                                      input.points,
                                      input.theta)
    end
end

function jacobian(j)
    input = GradBench.HT.input_from_json(j)
    complicated = size(input.us, 1) != 0
    if complicated
        wrapper = theta -> GradBench.HT.objective_complicated(input.model, input.correspondences, input.points, theta, input.us)
        y, jacobian_theta = Zygote.forward_jacobian(wrapper, input.theta)
        ylen = size(y, 1)
        jacobian_us = hcat([
            begin
                _, ju = Zygote.forward_jacobian(u -> GradBench.HT.objective_complicated(input.model, input.correspondences, input.points, input.theta, vcat(input.us[1:j-1], [u], input.us[j+1:end])), input.us[j])
                ju[:, 3j-2:3j]
            end
            for j ∈ 1:size(input.us, 1)
                ]...)
        vcat(jacobian_us, jacobian_theta)
    else
        Zygote.forward_jacobian(theta -> GradBench.HT.objective_simple(input.model, input.correspondences, input.points, theta), input.theta)[2]
    end
end

GradBench.register!("ht", Dict(
    "objective" => objective,
    "jacobian" => jacobian
))

end
