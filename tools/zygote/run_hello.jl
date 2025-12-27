module Hello

import Zygote
import GradBench

struct GradientHello <: GradBench.Hello.AbstractHello end
function (::GradientHello)(x)
    z, = Zygote.gradient(GradBench.Hello.square, x)
    return z
end

GradBench.register!(
    "hello", Dict(
        "square" => GradBench.Hello.PrimalHello(),
        "double" => GradientHello()
    )
)

end # module
