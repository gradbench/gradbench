module Hello

using ADTypes: AutoReverseDiff
import ReverseDiff
import GradBench

backend = AutoReverseDiff(; compile=true)

GradBench.register!("hello", Dict(
    "square" => GradBench.Hello.PrimalHello(),
    "double" => GradBench.Hello.DIGradientHello(backend)
))

end # module
