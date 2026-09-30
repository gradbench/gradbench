module KMeans

using ADTypes: AutoForwardDiff
import ForwardDiff
import GradBench

backend = AutoForwardDiff()

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.Pure.PrimalKMeans(),
        "dir" => GradBench.KMeans.DINewtonKMeans(GradBench.KMeans.Pure.PrimalKMeans(), backend)
    )
)

end # module
