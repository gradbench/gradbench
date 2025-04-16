module Hello

function square(x)
    return x * x
end

precompile(square, (Float64,))

using ADTypes: AbstractADType
import DifferentiationInterface as DI

function double(backend::AbstractADType, x::Real)
    return DI.derivative(square, backend, x)
end

end
