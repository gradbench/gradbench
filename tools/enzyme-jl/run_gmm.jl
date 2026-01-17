module GMM

using Enzyme
import GradBench

struct JacobianGMM <: GradBench.GMM.AbstractGMM end
function (::JacobianGMM)(input)
    k = size(input.means, 2)
    d = size(input.x, 1)
    Qs = GradBench.GMM.get_Qs(input.icfs, k, d)

    alpha_d, mu_d, Qs_d =
        Enzyme.gradient(
        set_runtime_activity(Reverse),
        GradBench.GMM.objective,
        input.alphas, input.means, Qs,
        Const(input.x), Const(input.wishart)
    )

    q_d, l_d = GradBench.GMM.Qs_to_q_l(d, Qs_d)

    return Dict(
        "alpha" => alpha_d,
        "mu" => mu_d,
        "q" => q_d,
        "l" => l_d
    )
end

GradBench.register!(
    "gmm", Dict(
        "objective" => GradBench.GMM.ObjectiveGMM(),
        "jacobian" => JacobianGMM()
    )
)

end # module
