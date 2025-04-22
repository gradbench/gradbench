module LLSQ

export Input, primal

struct Input
    x::Vector{Float64}
    n::Int64
end

function t(i::Int64, n::Int64)
    return -1 + (Float64(i) * 2) / (Float64(n) - 1)
end

function primal(x::Vector{T}, n::Int64) where {T}
    m = length(x)

    f(i) = begin
        ti = t(i, n)
        sum_g = sum(-x[j] * ti^Float64(j - 1) for j in 1:m)
        return (sign(ti) + sum_g)^2
    end

    return sum(f(i) for i in 0:n-1) / 2
end

end
