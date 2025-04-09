module Hello

using ADTypes: AbstractADType
import DifferentiationInterface as DI

function square(x)
    return x * x
end

function double(x, backend::AbstractADType)
    return DI.derivative(square, backend, x)
end

precompile(square, (Float64,))

end
