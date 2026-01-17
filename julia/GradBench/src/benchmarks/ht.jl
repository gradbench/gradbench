# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Based on: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteHT.jl

module HT

import GradBench
using LinearAlgebra

struct Model
    bone_names::Vector{String}
    parents::Vector{Int}
    base_relatives::Vector{Matrix{Float64}}
    inverse_base_absolutes::Vector{Matrix{Float64}}
    base_positions::Matrix{Float64}
    weights::Matrix{Float64}
    triangles::Vector{Vector{Int}}
    is_mirrored::Bool
end

struct Input
    model::Model
    correspondences::Vector{Int}
    points::Matrix{Float64}
    theta::Vector{Float64}
    "Has nonzero size only for 'complicated' kind of problems."
    us::Matrix{Float64}
end

abstract type AbstractHT <: GradBench.Experiment end

"Turn a vector of rows into the corresponding matrix."
function vecvecToMatrix(v::Vector{Vector{Float64}})::Matrix{Float64}
    return transpose(reduce(hcat, v))
end

function GradBench.preprocess(::AbstractHT, input)
    theta = input["theta"]
    us = input["us"]

    if size(us, 1) == 0
        us = Matrix{Float64}(undef, 0, 0)
    else
        us = vecvecToMatrix(convert(Vector{Vector{Float64}}, us))
    end

    correspondences = input["data"]["correspondences"]
    points = vecvecToMatrix(
        convert(
            Vector{Vector{Float64}},
            input["data"]["points"]
        )
    )

    base_positions = vecvecToMatrix(
        convert(
            Vector{Vector{Float64}},
            input["data"]["model"]["base_positions"]
        )
    )
    weights = vecvecToMatrix(
        convert(
            Vector{Vector{Float64}},
            input["data"]["model"]["weights"]
        )
    )
    triangles = input["data"]["model"]["triangles"]
    base_relatives = map(
        v -> vecvecToMatrix(convert(Vector{Vector{Float64}}, v)),
        input["data"]["model"]["base_relatives"]
    )
    inverse_base_absolutes = map(
        v -> vecvecToMatrix(convert(Vector{Vector{Float64}}, v)),
        input["data"]["model"]["inverse_base_absolutes"]
    )

    model = Model(
        input["data"]["model"]["bone_names"],
        input["data"]["model"]["parents"] .+ 1, # Julia indexing
        base_relatives,
        inverse_base_absolutes,
        transpose(base_positions),
        transpose(weights),
        [ triangles[i] .+ 1 for i in 1:size(triangles, 1) ], # Julia indexing
        input["data"]["model"]["is_mirrored"]
    )

    return (
        Input(
            model,
            correspondences .+ 1, # Julia indexing
            transpose(points),
            theta,
            us
        ),
    )
end

# objective
function angle_axis_to_rotation_matrix(angle_axis::Vector{T1})::Matrix{T1} where {T1}
    n = sqrt(sum(abs2, angle_axis))
    if n < 0.0001
        return Matrix{T1}(I, 3, 3)
    end

    x = angle_axis[1] / n
    y = angle_axis[2] / n
    z = angle_axis[3] / n

    s = sin(n)
    c = cos(n)

    return [
        x * x + (1 - x * x) * c x * y * (1 - c) - z * s x * z * (1 - c) + y * s;
        x * y * (1 - c) + z * s y * y + (1 - y * y) * c y * z * (1 - c) - x * s;
        x * z * (1 - c) - y * s z * y * (1 - c) + x * s z * z + (1 - z * z) * c
    ]
end

function apply_global_transform(pose_params::Vector{Vector{T1}}, positions::Matrix{T2})::Matrix{T2} where {T1, T2}
    return (angle_axis_to_rotation_matrix(pose_params[1]) .* pose_params[2]') * positions .+ pose_params[3]
end

function relatives_to_absolutes(relatives::Vector{Matrix{T1}}, parents::Vector{Int})::Vector{Matrix{T1}} where {T1}
    # Zygote does not support array mutation and on every iteration we may need to access
    # random element created on one of the previous iterations, so, no way to rewrite this
    # as a comprehension. Hence looped vcat.
    absolutes = Vector{Matrix{T1}}(undef, 0)
    for i in 1:length(parents)
        if parents[i] == 0
            absolutes = vcat(absolutes, [relatives[i]])
        else
            absolutes = vcat(absolutes, [absolutes[parents[i]] * relatives[i]])
        end
    end
    return absolutes
end

function euler_angles_to_rotation_matrix(xyz::Vector{T1})::Matrix{T1} where {T1}
    tx = xyz[1]
    ty = xyz[2]
    tz = xyz[3]
    costx = cos(tx)
    sintx = sin(tx)
    costy = cos(ty)
    sinty = sin(ty)
    costz = cos(tz)
    sintz = sin(tz)
    # We could define this as a 3x3 matrix and then build a block-diagonal
    # 4x4 matrix with 1. at (4, 4), but Zygote couldn't differentiate
    # any way of building that I could come up with.
    Rx = [ 1.0 0.0 0.0 0.0; 0.0 costx -sintx 0.0; 0.0 sintx costx 0.0; 0.0 0.0 0.0 1.0 ]
    Ry = [ costy 0.0 sinty 0.0; 0.0 1.0 0.0 0.0; -sinty 0.0 costy 0.0; 0.0 0.0 0.0 1.0 ]
    Rz = [ costz -sintz 0.0 0.0; sintz costz 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0 ]
    return Rz * Ry * Rx
end

function get_posed_relatives(model::Model, pose_params::Vector{Vector{T1}})::Vector{Matrix{T1}} where {T1}
    # default parametrization xzy # Flexion, Abduction, Twist
    order = [1, 3, 2]
    offset = 3
    n_bones = size(model.bone_names, 1)
    return [
        model.base_relatives[i_bone] * euler_angles_to_rotation_matrix(pose_params[i_bone + offset][order])
            for i_bone in 1:n_bones
    ]
end

function get_skinned_vertex_positions(model::Model, pose_params::Vector{Vector{T1}}, apply_global::Bool = true)::Matrix{T1} where {T1}
    relatives = get_posed_relatives(model, pose_params)
    absolutes = relatives_to_absolutes(relatives, model.parents)

    transforms = [ absolutes[i] * model.inverse_base_absolutes[i] for i in 1:size(absolutes, 1) ]

    n_verts = size(model.base_positions, 2)
    positions = zeros(Float64, 3, n_verts)
    for i in 1:size(transforms, 1)
        positions = positions +
            (transforms[i][1:3, :] * model.base_positions) .* model.weights[i, :]'
    end

    if model.is_mirrored
        positions = [-positions[1, :]'; positions[2:end, :]]
    end

    if apply_global
        positions = apply_global_transform(pose_params, positions)
    end
    return positions
end

function to_pose_params(theta::Vector{T1}, n_bones::Int)::Vector{Vector{T1}} where {T1}
    # fixed order pose_params
    #       1) global_rotation 2) scale 3) global_translation
    #       4) wrist
    #       5) thumb1, 6)thumb2, 7) thumb3, 8) thumb4
    #       similarly: index, middle, ring, pinky
    #       end) forearm

    n = 3 + n_bones
    n_fingers = 5
    cols = 5 + n_fingers * 4
    return [
        if i == 1
                theta[1:3]
        elseif i == 2
                [1.0, 1.0, 1.0]
        elseif i == 3
                theta[4:6]
        elseif i > cols || i == 4 || i % 4 == 1
                [0.0, 0.0, 0.0]
        elseif i % 4 == 2
                [theta[i + 1], theta[i + 2], 0.0]
        else
                [theta[i + 2], 0.0, 0.0]
        end
            for i in 1:n
    ]
end

function objective_simple(model::Model, correspondences::Vector{Int}, points::Matrix{T1}, theta::Vector{T2}) where {T1, T2}
    pose_params = to_pose_params(theta, length(model.bone_names))

    vertex_positions = get_skinned_vertex_positions(model, pose_params)

    n_corr = length(correspondences)
    return vcat([ points[:, i] - vertex_positions[:, correspondences[i]] for i in 1:n_corr ]...)
end

function objective_complicated(
        model::Model,
        correspondences::Vector{Int},
        points::Matrix{T1},
        theta::Vector{T2},
        us::Matrix{T3}
    ) where {T1, T2, T3}
    pose_params = to_pose_params(theta, length(model.bone_names))
    vertex_positions = get_skinned_vertex_positions(model, pose_params)

    verts_all = model.triangles[correspondences]

    # Extract the vertex indices as separate vectors
    verts1 = getindex.(verts_all, 1)
    verts2 = getindex.(verts_all, 2)
    verts3 = getindex.(verts_all, 3)

    # Compute the hand points in vectorized form
    hand_points = vertex_positions[:, verts1] .* us[:, 1]' .+
        vertex_positions[:, verts2] .* us[:, 2]' .+
        vertex_positions[:, verts3] .* (1 .- us[:, 1] .- us[:, 2])'

    # Compute the residuals and flatten to a vector
    residuals = points - hand_points
    return vec(residuals)
end

struct ObjectiveHT <: HT.AbstractHT end
function (::ObjectiveHT)(input)
    complicated = size(input.us, 1) != 0
    return if complicated
        objective_complicated(
            input.model,
            input.correspondences,
            input.points,
            input.theta,
            input.us
        )
    else
        objective_simple(
            input.model,
            input.correspondences,
            input.points,
            input.theta
        )
    end
end

end
