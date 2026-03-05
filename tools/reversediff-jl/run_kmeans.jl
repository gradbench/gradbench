module KMeans

using ADTypes: AutoReverseDiff
import ReverseDiff
import GradBench

backend = AutoReverseDiff(; compile = true)

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.PrimalKMeans(),
        "dir" => GradBench.KMeans.DINewtonKMeans(backend)
    )
)

end # module
