module ODE

using Enzyme
import GradBench

# This wraps 'GradBench.ODE.primal' with an 'out' parameter, instead
# of just returning the value, because Enzyme.jl cannot handle
# non-scalar return values.
function wrap(x::Vector{Float64}, s::Int, out::Vector{Float64})
    out .= GradBench.ODE.primal(x, s)
end

function primal(message)
    # TODO: move the message parsing into the harness
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    return GradBench.ODE.primal(x, s)
end

function gradient(message)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    output = similar(x)

    adj = Enzyme.make_zero(output)
    adj[end] = 1

    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, wrap, Const,
        Duplicated(x, dx),
        Const(s),
        Duplicated(output, adj)
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
