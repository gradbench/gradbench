# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Based on: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteGMM.jl

module GMM
using SpecialFunctions
using LinearAlgebra

export Wishart, GMMInput, input_from_json, objective, get_Qs, expdiags, diagsums, pack_J

struct Wishart
    gamma::Float64
    m::Int
end

struct GMMInput
    alphas::Matrix{Float64}
    means::Matrix{Float64}
    icfs::Matrix{Float64}
    x::Matrix{Float64}
    wishart::Wishart
end

function input_from_json(j)
    alphas = transpose(convert(Vector{Float64}, j["alpha"]))
    means = reduce(hcat, convert(Vector{Vector{Float64}},j["means"]))
    icfs = reduce(hcat, j["icf"])
    x = reduce(hcat, j["x"])
    gamma = convert(Float64, j["gamma"])
    m = convert(Int, j["m"])

    return GMMInput(alphas, means, icfs, x, Wishart(gamma, m))
end

"Computes logsumexp. Input should be 1 dimensional"
function logsumexp(x)
    mx = maximum(x)
    log(sum(exp.(x .- mx))) + mx
end

function ltri_unpack(D, LT)
    d = length(D)
    make_col(r::Int, L) = vcat(zeros(r - 1), D[r], reshape([L[i] for i=1:d-r], d - r))
    col_start(r::Int) = (r - 1) * (2d - r) ÷ 2
    inds(r) = col_start(r) .+ (1:d-r)
    hcat([make_col(r, LT[inds(r)]) for r=1:d]...)
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

function unzip(tuples)
    map(1:length(first(tuples))) do i
        map(tuple -> tuple[i], tuples)
    end
end

Base.:*(::Float64, ::Nothing) = nothing

# This function requires an argument 'Qs' that is not immediately part
# of GMMInput. Instead it must be extracted from 'icfs' using the
# function 'get_Qs'. This is somewhat different to our other
# implementations of GMM, where the extraction of Qs is done inside
# the objective function itself. I believe the cause is that 'get_Qs'
# is not handled well by some of the Julia AD tools.
function objective(alphas, means, Qs, x, wishart::Wishart)
    d = size(x, 1)
    n = size(x, 2)
    k = size(means, 2)
    CONSTANT = -n * d * 0.5 * log(2 * pi)
    sum_qs = reshape(diagsums(Qs), 1, size(Qs, 3))
    Qs = expdiags(Qs)

    slse = 0.
    for ix=1:n
        formula(ik) = -0.5 * sum(abs2, Qs[:, :, ik] * (x[:,ix] .- means[:, ik])) + alphas[ik] + sum_qs[ik]
        terms = map(formula, 1:k)
        slse += logsumexp(terms)
    end

    CONSTANT + slse - n * logsumexp(alphas) + log_wishart_prior(wishart, sum_qs, Qs, k)
end

# The objective function is defined in terms of Qs, which are
# extracted from icfs. This means the Jacobian doesn't look exactly
# how it is supposed to. This function packs the Jacobian
# appropriately.
function pack_J(J, k, d)
    alphas = reshape(J[1], :)
    means = reshape(J[2], :)
    icf_unpacked = map(1:k) do Q_idx
        Q = J[3][:, :, Q_idx]
        lt_cols = map(1:d-1) do col
            Q[col+1:d, col]
        end
        vcat(diag(Q), lt_cols...)
    end
    icf = collect(Iterators.flatten(icf_unpacked))
    packed_J = vcat(alphas, means, icf)
    packed_J
end

end
