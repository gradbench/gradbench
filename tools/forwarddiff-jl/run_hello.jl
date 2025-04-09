module Hello

import DifferentiationInterface as DI
import ForwardDiff
import GradBench

function double(x)
    z, = DI.derivative(GradBench.Hello.square, DI.AutoForwardDiff(), x)
    return z
end

precompile(double, (Float64,))

GradBench.register!(
    "hello", Dict(
        "square" => GradBench.Hello.square,
        "double" => double
    )
)

end # module
