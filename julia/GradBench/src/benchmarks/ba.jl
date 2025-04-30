# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Based on: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteBA.jl

module BA

using LinearAlgebra
import ..GradBench

const N_CAM_PARAMS = 11
const ROT_IDX = 1
const C_IDX = 4
const F_IDX = 7
const X0_IDX = 8
const RAD_IDX = 10

struct BAInput
    n::Int
    m::Int
    p::Int
    cams::Matrix{Float64}
    X::Matrix{Float64}
    w::Vector{Float64}
    feats::Matrix{Float64}
    obs::Matrix{Int}
end

abstract type AbstractBA <: GradBench.Experiment end

function GradBench.preprocess(::AbstractBA, input)
    n = input["n"]
    m = input["m"]
    p = input["p"]

    one_cam = convert(Vector{Float64}, input["cam"])
    one_X = convert(Vector{Float64}, input["x"])
    one_w = input["w"]
    one_feat = convert(Vector{Float64}, input["feat"])

    cams = repeat(one_cam, 1, n)
    X = repeat(one_X, 1, m)
    w = repeat([one_w], p)
    feats = repeat(one_feat, 1, p)

    camIdx = 1
    ptIdx = 1
    obs = zeros(Int, 2, p)
    for i in 1:p
        obs[1, i] = camIdx
        obs[2, i] = ptIdx
        camIdx = (camIdx % n) + 1
        ptIdx = (ptIdx % m) + 1
    end

    return (BAInput(n, m, p, cams, X, w, feats, obs),)
end

struct SparseMatrix
    "Number of cams"
    n::Int
    "Number of points"
    m::Int
    "Number of observations"
    p::Int
    nrows::Int
    ncols::Int
    """
    Int[nrows + 1]. Defined recursively as follows:
    rows[0] = 0
    rows[i] = rows[i-1] + the number of nonzero elements on the i-1 row of the matrix
    """
    rows::Vector{Int}
    "Column index in the matrix of each element of vals. Has the same size"
    cols::Vector{Int}
    "All the nonzero entries of the matrix in the left-to-right top-to-bottom order"
    vals::Vector{Float64}
    SparseMatrix(n::Int, m::Int, p::Int) = new(n, m, p, 2 * p + p, N_CAM_PARAMS * n + 3 * m + p, [0], [], [])
end

function insert_reproj_err_block!(matrix::SparseMatrix, obsIdx::Int, camIdx::Int, ptIdx::Int, J::AbstractMatrix{Float64})
    # We use zero-based indexing for storage, but Julia uses 1-bsed indexing
    # Hence, the conversion
    obsIdxZeroBased = obsIdx - 1
    camIdxZeroBased = camIdx - 1
    ptIdxZeroBased = ptIdx - 1
    n_new_cols = N_CAM_PARAMS + 3 + 1
    lastrow = matrix.rows[end]
    push!(matrix.rows, lastrow + n_new_cols, lastrow + n_new_cols + n_new_cols)
    for i_row ∈ 1:2
        for i ∈ 1:N_CAM_PARAMS
            push!(matrix.cols, N_CAM_PARAMS * camIdxZeroBased + (i - 1))
            push!(matrix.vals, J[i_row, i])
        end
        col_offset = N_CAM_PARAMS * matrix.n
        for i ∈ 1:3
            push!(matrix.cols, col_offset + 3 * ptIdxZeroBased + (i - 1))
            push!(matrix.vals, J[i_row, N_CAM_PARAMS+i])
        end
        col_offset += 3 * matrix.m
        val_offset = N_CAM_PARAMS + 3
        push!(matrix.cols, col_offset + obsIdxZeroBased)
        push!(matrix.vals, J[i_row, val_offset+1])
    end
end

function insert_w_err_block!(matrix::SparseMatrix, wIdx::Int, w_d::Float64)
    # We use zero-based indexing for storage, but Julia uses 1-bsed indexing
    # Hence, the conversion
    wIdxZeroBased = wIdx - 1
    push!(matrix.rows, matrix.rows[end] + 1)
    push!(matrix.cols, N_CAM_PARAMS * matrix.n + 3 * matrix.m + wIdxZeroBased)
    push!(matrix.vals, w_d)
end

# The BA input is duplicated during unpacking, so we deduplicate the
# Jacobian at the end.
function dedup_jacobian(J)
    Dict("rows" => vcat(J.rows[1:30], [J.rows[end]]),
        "cols" => vcat(J.cols[1:30], [J.cols[end]]),
        "vals" => vcat(J.vals[1:30], [J.vals[end]]))
end

function rodrigues_rotate_point(rot::Vector{T}, X::Vector{T}) where {T}
    sqtheta = sum(rot .* rot)
    if sqtheta > 1e-10
        theta = sqrt(sqtheta)
        costheta = cos(theta)
        sintheta = sin(theta)
        theta_inverse = 1.0 / theta

        w = theta_inverse * rot
        w_cross_X = cross(w, X)
        tmp = dot(w, X) * (1.0 - costheta)

        X * costheta + w_cross_X * sintheta + w * tmp
    else
        X + cross(rot, X)
    end
end

function radial_distort(rad_params, proj)
    rsq = sum(proj .* proj)
    L = 1.0 + rad_params[1] * rsq + rad_params[2] * rsq * rsq
    proj * L
end

function project(cam, X)
    Xcam = rodrigues_rotate_point(cam[ROT_IDX:ROT_IDX+2], X - cam[C_IDX:C_IDX+2])
    distorted = radial_distort(cam[RAD_IDX:RAD_IDX+1], Xcam[1:2] / Xcam[3])
    distorted * cam[F_IDX] + cam[X0_IDX:X0_IDX+1]
end

function compute_reproj_err(cam, X, w, feat)
    w * (project(cam, X) - feat)
end

function objective(cams, X, w, obs, feats)
    reproj_err = similar(feats)
    for i in 1:size(feats, 2)
        reproj_err[:, i] = compute_reproj_err(cams[:, obs[1, i]], X[:, obs[2, i]], w[i], feats[:, i])
    end
    w_err = 1.0 .- w .* w
    (reproj_err, w_err)
end

struct ObjectiveBA <: BA.AbstractBA end
function (::ObjectiveBA)(input)
    (r_err, w_err) =
        objective(input.cams,
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


end
