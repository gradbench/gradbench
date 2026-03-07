module KMeans

using ADTypes: AutoMooncake, AutoMooncakeForward
using DifferentiationInterface: SecondOrder
import Mooncake
import GradBench

backend = SecondOrder(AutoMooncakeForward(), AutoMooncake())

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.Pure.PrimalKMeans(),
        "dir" => GradBench.KMeans.DINewtonKMeans(GradBench.KMeans.Pure.PrimalKMeans(), backend)
    )
)

end # module
