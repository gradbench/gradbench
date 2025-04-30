module Det

using Enzyme
import GradBench

struct GradientDet <: GradBench.Det.AbstractDet end
function (::GradientDet)(A, ell)
    M = transpose(reshape(A, ell, ell))
    z, = Enzyme.gradient(Reverse, GradBench.Det.Impure.det_by_minor, M)
    return reshape(z', ell * ell)
end

GradBench.register!("det", Dict(
    "primal" => GradBench.Det.Impure.PrimalDet(),
    "gradient" => GradientDet()
))


end
