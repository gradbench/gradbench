module Hello

using Enzyme
import GradBench

function double(x)
    z, = Enzyme.gradient(Reverse, GradBench.Hello.square, x)
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
