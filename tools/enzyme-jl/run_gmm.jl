module GMM

using Enzyme
import GradBench

struct JacobianGMM <: GradBench.GMM.AbstractGMM end
function (::JacobianGMM)(input)
    k = size(input.means, 2)
    d = size(input.x, 1)
    Qs = GradBench.GMM.get_Qs(input.icfs, k, d)

    J =
        Enzyme.gradient(set_runtime_activity(Reverse),
                        GradBench.GMM.objective,
                        input.alphas, input.means, Qs,
                        Const(input.x), Const(input.wishart))

    GradBench.GMM.pack_J(J, k, d)
end

GradBench.register!("gmm", Dict(
    "objective" => GradBench.GMM.ObjectiveGMM(),
    "jacobian" => JacobianGMM()
))

end # module
