module Hello

using Enzyme
import GradBench

struct GradientHello <: GradBench.Hello.AbstractHello end

function (::GradientHello)(x)
    z, = Enzyme.gradient(Reverse, GradBench.Hello.square, x)
    return z
end

GradBench.register!(
    "hello", Dict(
        "square" => GradBench.Hello.PrimalHello(),
        "double" => GradientHello(),
    )
)

end # module
