module KMeans

using ADTypes: AutoForwardDiff
import ForwardDiff
import GradBench

backend = AutoForwardDiff()

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.PrimalKMeans(),
        "dir" => GradBench.KMeans.DINewtonKMeans(backend)
    )
)

end # module
