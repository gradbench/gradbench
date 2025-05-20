module GMM

import Zygote
import GradBench

Zygote.@adjoint function GradBench.GMM.diagsums(Qs)
    GradBench.GMM.diagsums(Qs),
    function (Δ)
        Δ′ = zero(Qs)
        for (i, δ) in enumerate(Δ)
            for j in 1:size(Qs, 1)
                Δ′[j, j, i] = δ
            end
        end
        (Δ′,)
    end
end

Zygote.@adjoint function GradBench.GMM.expdiags(Qs)
    GradBench.GMM.expdiags(Qs),
    function (Δ)
        Δ′ = zero(Qs)
        Δ′ .= Δ
        for i in 1:size(Qs, 3)
            for j in 1:size(Qs, 1)
                Δ′[j, j, i] *= exp(Qs[j, j, i])
            end
        end
        (Δ′,)
    end
end

struct JacobianGMM <: GradBench.GMM.AbstractGMM end
function (::JacobianGMM)(input)
    k = size(input.means, 2)
    d = size(input.x, 1)
    Qs = GradBench.GMM.get_Qs(input.icfs, k, d)

    function wrap(alpha, means, Qs)
        GradBench.GMM.objective(alpha, means, Qs, input.x, input.wishart)
    end

    alpha_d, mu_d, Qs_d = Zygote.gradient(wrap, input.alphas, input.means, Qs)

    # TODO: extract q_d, l_d from Qs_d.

    return Dict("alpha" => alpha_d,
                "mu" => mu_d,
                "q" => false,
                "l" => false)
end


GradBench.register!("gmm", Dict(
    "objective" => GradBench.GMM.ObjectiveGMM(),
    "jacobian" => JacobianGMM()
))

end
