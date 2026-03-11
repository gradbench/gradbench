module KMeans

import ..GradBench
using ..ADTypes: AbstractADType
import ..DifferentiationInterface as DI

abstract type AbstractKMeans <: GradBench.Experiment end

struct KMeansInput
    points::Matrix{Float64}
    centroids::Matrix{Float64}
end

function GradBench.preprocess(::AbstractKMeans, input)
    points = stack(convert(Vector{Vector{Float64}}, input["points"]); dims = 1)
    centroids = stack(convert(Vector{Vector{Float64}}, input["centroids"]); dims = 1)
    return (; input = KMeansInput(points, centroids))
end
module Pure
    using ..KMeans
    import ...DifferentiationInterface as DI

    function f(C::AbstractMatrix, P::AbstractMatrix)
        return sum(axes(P, 1)) do i
            Pi = view(P, i, :)
            minimum(axes(C, 1)) do j
                Cj = view(C, j, :)
                dist = mapreduce(abs2 ∘ -, +, Cj, Pi)
            end
        end
    end

    struct PrimalKMeans <: KMeans.AbstractKMeans end

    function (::PrimalKMeans)(input::KMeans.KMeansInput)
        return f(input.centroids, input.points)
    end
end

module Impure
    using ..KMeans
    import ...DifferentiationInterface as DI

    function f(C::AbstractMatrix, P::AbstractMatrix)
        T = eltype(C)
        total_cost = zero(T)
        @inbounds for i in axes(P, 1)
            min_dist = typemax(T)
            for j in axes(C, 1)
                dist = zero(T)
                for k in axes(C, 2)
                    diff = C[j, k] - P[i, k]
                    dist += diff * diff
                end
                if dist < min_dist
                    min_dist = dist
                end
            end
            total_cost += min_dist
        end
        return total_cost
    end

    struct PrimalKMeans <: KMeans.AbstractKMeans end

    function (::PrimalKMeans)(input::KMeans.KMeansInput)
        return f(input.centroids, input.points)
    end
end

struct DINewtonKMeans{P, B <: AbstractADType} <: GradBench.Experiment
    primal::P
    backend::B
end

function GradBench.preprocess(n::DINewtonKMeans, message)
    (; primal, backend) = n
    (; input) = GradBench.preprocess(primal, message)
    (; centroids, points) = input
    prep = DI.prepare_hvp(primal, backend, centroids, (zero(centroids),), DI.Constant(points))
    return (; prep, input)
end

function (n::DINewtonKMeans)(prep, input)
    (; primal, backend) = n
    (; centroids, points) = input
    dcentroids = fill!(similar(centroids), one(eltype(centroids)))
    J, Hs = DI.gradient_and_hvp(primal, prep, backend, centroids, (dcentroids,), DI.Constant(points))
    H = only(Hs)
    dir = J .* inv.(H)
    return collect(eachrow(dir))
end

end
