module KMeans

import Enzyme
import GradBench

function kmeans_objective_d(C, dC, P)
    Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(GradBench.KMeans.Impure.f), Enzyme.Active, Enzyme.Duplicated(C, dC), Enzyme.Const(P))
    return nothing
end

struct GradientKMeans <: GradBench.KMeans.AbstractKMeans end

function GradBench.preprocess(n::GradientKMeans, message)
    (; primal) = n
    (; input) = GradBench.preprocess(primal, message)
    (; centroids, points) = input
    prep = (; C_seed = ones(C), J = similar(centroids), H = similar(centroids))
    return (; prep, input)
end

function (::GradientKMeans)(prep, input::GradBench.KMeans.KMeansInput)
    (; C_seed, J, H) = prep
    C = input.centroids
    P = input.points

    Enzyme.autodiff(Enzyme.Forward, Enzyme.Const(kmeans_objective_d), Enzyme.Duplicated(C, C_seed), Enzyme.Duplicated(J, H), Enzyme.Const(P))

    dir = similar(J, length(J))
    @inbounds for i in 1:length(J)
        dir[i] = J[i] / H[i]
    end
    return dir
end

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.Impure.PrimalKMeans(),
        "dir" => GradientKMeans()
    )
)

end # module
