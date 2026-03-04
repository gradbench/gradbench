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

function f(C::AbstractMatrix, P::AbstractMatrix)
    return sum(axes(P, 1)) do i
        Pi = view(P, i, :)
        minimum(axes(C, 1)) do j
            Cj = view(C, j, :)
            dist = mapreduce(abs2 ∘ -, +, Cj, Pi)
        end
    end
end

struct PrimalKMeans <: AbstractKMeans end

function (::PrimalKMeans)(input::KMeansInput)
    return f(input.centroids, input.points)
end

struct DINewtonKMeans{B <: AbstractADType} <: GradBench.Experiment
    backend::B
end

function GradBench.preprocess(n::DINewtonKMeans, message)
    (; backend) = n
    (; input) = GradBench.preprocess(PrimalKMeans(), message)
    (; centroids, points) = input
    prep = DI.prepare_hvp(f, backend, centroids, (zero(centroids),), DI.Constant(points))
    return (; prep, input)
end

function (n::DINewtonKMeans)(prep, input)
    (; backend) = n
    (; centroids, points) = input
    dcentroids = fill!(similar(centroids), one(eltype(centroids)))
    J, Hs = DI.gradient_and_hvp(f, prep, backend, centroids, (dcentroids,), DI.Constant(points))
    H = only(Hs)
    dir = J .* inv.(H)
    return collect(eachrow(dir))
end

end
