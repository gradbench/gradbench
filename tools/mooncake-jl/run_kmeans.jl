module KMeans

using ADTypes: AutoMooncake
import Mooncake
import GradBench

backend = AutoMooncake()

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.PrimalKMeans(),
        "dir" => GradBench.KMeans.DINewtonKMeans(backend)
    )
)

end # module
