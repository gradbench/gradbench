module KMeans

using ADTypes: AutoReverseDiff
import ReverseDiff
import GradBench

backend = AutoReverseDiff(; compile = true)

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.Pure.PrimalKMeans(),
        "dir" => GradBench.KMeans.DINewtonKMeans(GradBench.KMeans.Pure.PrimalKMeans(), backend)
    )
)

end # module
