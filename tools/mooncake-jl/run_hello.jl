module Hello

using ADTypes: AutoMooncake
import Mooncake
import GradBench

GradBench.register!(
    "hello", Dict(
        "square" => GradBench.Hello.square,
        "double" => (GradBench.Hello.double, AutoMooncake(; config=nothing)),
    )
)

end # module
