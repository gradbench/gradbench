module Hello

using ADTypes: AutoMooncake
import Mooncake
import GradBench

backend = AutoMooncake(; config=nothing)

GradBench.register!("hello", Dict(
    "square" => GradBench.Hello.PrimalHello(),
    "double" => GradBench.Hello.DIGradientHello(backend)
))

end # module
