# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Based on: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteGMM.jl

module GMM

import GradBench
using SpecialFunctions
using LinearAlgebra

struct Wishart
    gamma::Float64
    m::Int
end

struct Input
    alphas::Matrix{Float64}
    means::Matrix{Float64}
    icfs::Matrix{Float64}
    x::Matrix{Float64}
    wishart::Wishart
end

abstract type AbstractGMM <: GradBench.Experiment end

function GradBench.preprocess(::AbstractGMM, j)
    alphas = transpose(convert(Vector{Float64}, j["alpha"]))
    means = reduce(hcat, convert(Vector{Vector{Float64}}, j["means"]))
    icfs = reduce(hcat, j["icf"])
    x = reduce(hcat, j["x"])
    gamma = convert(Float64, j["gamma"])
    m = convert(Int, j["m"])

    return (Input(alphas, means, icfs, x, Wishart(gamma, m)),)
end

"Computes logsumexp. Input should be 1 dimensional"
function logsumexp(x)
    mx = maximum(x)
    log(sum(exp.(x .- mx))) + mx
end

function ltri_unpack(D, LT)
    d = length(D)
    make_col(r::Int, L) = vcat(zeros(r - 1), D[r], reshape([L[i] for i = 1:d-r], d - r))
    col_start(r::Int) = (r - 1) * (2d - r) ÷ 2
    inds(r) = col_start(r) .+ (1:d-r)
    hcat([make_col(r, LT[inds(r)]) for r = 1:d]...)
end

function get_Q(d, icf)
    ltri_unpack((icf[1:d]), icf[d+1:end])
end

function get_Qs(icfs, k, d)
    cat([get_Q(d, icfs[:, ik]) for ik in 1:k]...;
        dims=[3])
end

function log_gamma_distrib(a, p)
    out = 0.25 * p * (p - 1) * 1.1447298858494002 #convert(Float64, log(pi))
    out += sum(j -> loggamma(a + 0.5 * (1 - j)), 1:p)
    out
end

function log_wishart_prior(wishart::Wishart, sum_qs, Qs, k)
    p = size(Qs, 1)
    n = p + wishart.m + 1
    C = n * p * (log(wishart.gamma) - 0.5 * log(2)) - log_gamma_distrib(0.5 * n, p)

    frobenius = sum(abs2, Qs)
    0.5 * wishart.gamma^2 * frobenius - wishart.m * sum(sum_qs) - k * C
end

function diagsums(Qs)
    mapslices(slice -> sum(diag(slice)), Qs; dims=[1, 2])
end

function expdiags(Qs)
    mapslices(Qs; dims=[1, 2]) do slice
        slice[diagind(slice)] .= exp.(slice[diagind(slice)])
        slice
    end
end

Base.:*(::Float64, ::Nothing) = nothing

# This function requires an argument 'Qs' that is not immediately part
# of Input. Instead it must be extracted from 'icfs' using the
# function 'get_Qs'. This is somewhat different to our other
# implementations of GMM, where the extraction of Qs is done inside
# the objective function itself. I believe the cause is that 'get_Qs'
# is not handled well by some of the Julia AD tools.
function objective(alphas, means, Qs, x, wishart::Wishart)
    d, n = size(x)
    k = size(means, 2)
    CONSTANT = -n * d * 0.5 * log(2π)

    # Precompute
    sum_qs = reshape(diagsums(Qs), 1, size(Qs, 3))  # 1 × k
    Qs = expdiags(Qs)  # d × d × k

    diffs = reshape(x, d, 1, n) .- reshape(means, d, k, 1)  # d × k × n
    Q_diffs = map(i -> Qs[:, :, i] * diffs[:, i, :], 1:k)  # list of d × n
    Q_diffs = reshape(hcat(Q_diffs...), d, n, k)  # d × n × k
    norms = sum(abs2, Q_diffs; dims=1)  # 1 × n × k
    norms = reshape(norms, n, k)  # n × k

    # Compute log-terms for each data point and cluster
    log_terms = -0.5 * norms .+ reshape(alphas, 1, :) .+ repeat(sum_qs, n, 1)  # n × k

    # Apply logsumexp row-wise
    slse = sum(logsumexp, eachrow(log_terms))

    CONSTANT + slse - n * logsumexp(alphas) + log_wishart_prior(wishart, sum_qs, Qs, k)
end

struct ObjectiveGMM <: GMM.AbstractGMM end
function (::ObjectiveGMM)(input)
    k = size(input.means, 2)
    d = size(input.x, 1)
    Qs = GradBench.GMM.get_Qs(input.icfs, k, d)

    return objective(input.alphas, input.means, Qs, input.x, input.wishart)
end

# The objective function is defined in terms of Qs, which are
# extracted from icfs. This means the Jacobian doesn't look exactly
# how it is supposed to. This function packs the Jacobian
# appropriately.
function pack_J(J, k, d)
    alphas = vec(J[1])
    means = vec(J[2])
    icf = Vector{Float64}(undef, k * (d + (d * (d - 1)) ÷ 2))

    idx = 1
    for Q_idx in 1:k
        Q = J[3][:, :, Q_idx]
        icf[idx:idx+d-1] = diag(Q)
        idx += d
        for col in 1:d-1
            len = d - col
            icf[idx:idx+len-1] = @view Q[col+1:d, col]
            idx += len
        end
    end

    return vcat(alphas, means, icf)
end

end
