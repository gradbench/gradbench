module Hello

using ADTypes: AbstractADType
import DifferentiationInterface as DI

function square(x::Real)
    return x * x
end

function double(backend::AbstractADType, x::Real)
    return DI.derivative(square, backend, x)
end

precompile(square, (Float64,))

end
