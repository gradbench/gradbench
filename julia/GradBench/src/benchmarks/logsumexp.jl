module LogSumExp

struct Input
    x::Vector{Float64}
end

function logsumexp(x::Vector{T}) where {T}
    xmax = maximum(x)
    return xmax + log(sum(exp.(x .- xmax)))
end

function primal(input)
    x = convert(Vector{Float64}, input["x"])
    return logsumexp(x)
end

end # module logsumexp
