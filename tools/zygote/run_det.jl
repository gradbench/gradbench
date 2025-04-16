module Det

import Zygote
import GradBench

function gradient(j)
    input = GradBench.Det.input_from_json(j)
    M = transpose(reshape(input.A, input.ell, input.ell))
    z, = Zygote.gradient(GradBench.Det.det_by_minor, M)
    return reshape(z', input.ell*input.ell)
end

GradBench.register!("det", Dict(
    "primal" => GradBench.Det.primal ∘ GradBench.Det.input_from_json,
    "gradient" => gradient
))


end
