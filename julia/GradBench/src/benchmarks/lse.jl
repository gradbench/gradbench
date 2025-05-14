module LSE

import ..GradBench

struct Input
    x::Vector{Float64}
end

abstract type AbstractLSE <: GradBench.Experiment end

function GradBench.preprocess(::AbstractLSE, message)
    (Input(convert(Vector{Float64}, message["x"])),)
end

# A sequential implementation in a pure and vectorised style.
module Pure

using ..LSE

function logsumexp(x::Vector{T}) where {T}
    xmax = maximum(x)
    return xmax + log(sum(exp.(x .- xmax)))
end

struct PrimalLSE <: LSE.AbstractLSE end
(::PrimalLSE)(input) = logsumexp(input.x)

end # Pure

# A multithreaded implementation that uses side effects.
module Impure

using ..LSE

using Base.Threads

"Explicitly loopy with @threads annotations for parallel execution."
function logsumexp(x::Vector{T}) where {T}
    n = length(x)
    nt = nthreads()

    local_max = fill(x[1], nt)
    Threads.@threads for i in 1:n
        tid = threadid()
        local_max[tid] = max(local_max[tid], x[i])
    end
    xmax = maximum(local_max)

    local_sum = zeros(T, nt)
    Threads.@threads for i in 1:n
        tid = threadid()
        local_sum[tid] += exp(x[i] - xmax)
    end
    total = sum(local_sum)

    return xmax + log(total)
end

struct PrimalLSE <: LSE.AbstractLSE end
(::PrimalLSE)(input) = logsumexp(input.x)

end # module Impure

end # module lse
