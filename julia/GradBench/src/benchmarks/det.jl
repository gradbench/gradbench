# This module provides two implementations: one that requires
# mutation, and one that does not.
module Det

import ..GradBench
using ..ADTypes: AbstractADType
import ..DifferentiationInterface as DI

abstract type AbstractDet <: GradBench.Experiment end

function GradBench.preprocess(::AbstractDet, message)
    A = convert(Vector{Float64}, message["A"])
    ell = message["ell"]
    (; A, ell)
end

module Pure

using ..Det

function minor(matrix::AbstractMatrix{T}, row::Int, col::Int) where T
    matrix[[1:row-1; row+1:end], [1:col-1; col+1:end]]
end

function det_by_minor(matrix::AbstractMatrix{T}) where T
    n = size(matrix, 1)
    if n == 1
        return matrix[1, 1]
    elseif n == 2
        return matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]
    else
        det = zero(T)
        for col in 1:n
            sign = (-1)^(1 + col)
            sub_det = det_by_minor(minor(matrix, 1, col))
            det += sign * matrix[1, col] * sub_det
        end
        return det
    end
end

function primal(A, ell)
    return det_by_minor(transpose(reshape(A, ell, ell)))
end

struct PrimalDet <: Det.AbstractDet end
function (::PrimalDet)(A, ell)
    return primal(A, ell)
end

end # module Pure

# This is written in a very imperative style mirroring the one used
# for C++.
module Impure

using ..Det

function det_of_minor(A::AbstractMatrix{T},
    n::Int,
    m::Int,
    r::Vector{Int},
    c::Vector{Int}) where T
    R0 = r[n+1]
    @assert R0 <= n
    Cj = c[n+1]
    @assert Cj <= n

    if m == 1
        return A[R0, Cj]
    end

    detM = zero(T)
    sign = 1
    r[n+1] = r[R0+1]
    Cj1 = n

    for j in 1:m
        M0j = A[R0, Cj]

        c[Cj1+1] = c[Cj+1]
        detS = det_of_minor(A, n, m - 1, r, c)
        c[Cj1+1] = Cj

        if sign > 0
            detM += M0j * detS
        else
            detM -= M0j * detS
        end

        Cj1 = Cj
        Cj = c[Cj+1]
        sign = -sign
    end

    r[n+1] = R0
    return detM
end

function det_by_minor(A)
    ell = size(A, 1)
    r = Vector(1:ell+1)
    r[end] = 1
    c = Vector(1:ell+1)
    c[end] = 1
    return det_of_minor(A, ell, ell, r, c)
end

function primal(A, ell)
    return det_by_minor(transpose(reshape(A, ell, ell)))
end

struct PrimalDet <: Det.AbstractDet end
function (::PrimalDet)(A, ell)
    return primal(A, ell)
end

end # module Impure

struct DIGradientDet{P,B<:AbstractADType} <: GradBench.Experiment
    primal::P
    backend::B
end

function GradBench.preprocess(g::DIGradientDet, message)
    (; primal, backend) = g
    (; A, ell) = GradBench.preprocess(primal, message)
    prep = DI.prepare_gradient(primal, backend, zero(A), DI.Constant(ell))
    return (; prep, A, ell)
end

function (g::DIGradientDet)(prep, A, ell)
    (; primal, backend) = g
    return DI.gradient(primal, prep, backend, A, DI.Constant(ell))
end

end
