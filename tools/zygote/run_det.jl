module Det

import Zygote
import GradBench

struct GradientDet <: GradBench.Det.AbstractDet end
function (::GradientDet)(A, ell)
    z, = Zygote.gradient(A -> GradBench.Det.Pure.primal(A, ell)[end],
                         A)
    return z
end


function gradient(j)
    input = GradBench.Det.input_from_json(j)
    M = transpose(reshape(input.A, input.ell, input.ell))
    z, = Zygote.gradient(GradBench.Det.Pure.det_by_minor, M)
    return reshape(z', input.ell*input.ell)
end

GradBench.register!("det", Dict(
    "primal" => GradBench.Det.Pure.PrimalDet(),
    "gradient" => GradientDet()
))


end
