module ODE

using Enzyme
import GradBench

function primal(message)
    # TODO: move the message parsing into the harness
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    output = similar(x)

    GradBench.ODE.primal!(output, x, s)
    return output
end

function gradient(message)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    output = similar(x)

    adj = Enzyme.make_zero(output)
    adj[end] = 1

    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, GradBench.ODE.primal!, Const,
        Duplicated(output, adj),
        Duplicated(x, dx),
        Const(s),
    )
    return dx
end

GradBench.register!(
    "ode", Dict(
        "primal" => primal,
        "gradient" => gradient
    )
)

end # module
