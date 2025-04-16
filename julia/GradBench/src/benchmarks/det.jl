module Det

export Input, input_from_json, det_by_minor, primal

struct Input
    A::Vector{Float64}
    ell::Int
end

function input_from_json(j)
    Input(j["A"], j["ell"])
end

function minor(matrix::AbstractMatrix{T}, row::Int, col::Int) where T
    matrix[[1:row-1; row+1:end], [1:col-1; col+1:end]]
end

function det_by_minor(matrix::AbstractMatrix{T}) where T
    n = size(matrix, 1)
    if n == 1
        return matrix[1,1]
    elseif n == 2
        return matrix[1,1] * matrix[2,2] - matrix[1,2] * matrix[2,1]
    else
        det = zero(T)
        for col in 1:n
            sign = (-1)^(1 + col)
            sub_det = det_by_minor(minor(matrix, 1, col))
            det += sign * matrix[1,col] * sub_det
        end
        return det
    end
end

function primal(input::Input)
    return det_by_minor(transpose(reshape(input.A, input.ell, input.ell)))
end

end
