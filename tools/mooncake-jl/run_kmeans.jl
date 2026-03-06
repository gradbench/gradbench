module KMeans

using ADTypes: AutoMooncake, AutoMooncakeForward
using DifferentiationInterface: SecondOrder
import Mooncake
import GradBench

backend = SecondOrder(AutoMooncakeForward(), AutoMooncake())

GradBench.register!(
    "kmeans", Dict(
        "cost" => GradBench.KMeans.PrimalKMeans(),
        "dir" => GradBench.KMeans.DINewtonKMeans(backend)
    )
)

end # module
