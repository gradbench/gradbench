module KMeans

using ADTypes: AutoZygote
import Zygote
import GradBench

# Zygote doesn't have a built-in HVP so it was easier to go through DI, feel free to modify it to pure Zygote if necessary
backend = AutoZygote()

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.PrimalKMeans(),
        "dir" => GradBench.KMeans.DINewtonKMeans(backend)
    )
)

end # module
