import SciLean
import Gradbench.KMeans.Data
import Gradbench.Util

namespace Gradbench.KMeans

open SciLean Scalar

set_default_scalar Float

instance {n:ℕ} : Inhabited (Fin n) := ⟨0, sorry_proof⟩

set_option linter.unusedVariables false in
@[data_synth high]
theorem SciLean.sum.arg_f.HasRevFDeriv_rule_scalar
    {K} [RCLike K]
    {W} [NormedAddCommGroup W] [AdjointSpace K W]
    {I : Type*} [IndexType I]
    (f : W → I → K) {f' : I → _} (hf : ∀ i, HasRevFDerivUpdate K (f · i) (f' i))  :
    HasRevFDeriv K
      (fun w => sum (f w))
      (fun w =>
        let' (s,dw) := IndexType.foldl (init := ((0 : K), (0:W)))
          (fun sdw (i : I) =>
            let' (s,dw) := sdw
            let' (x,df) := f' i w
            let s := s + x
            (s, df 1 dw))
        (s, fun dx => dx•dw)) := sorry_proof


def kmeansObjective {n k d : ℕ} (points : Float^[d]^[n]) (centroids : Float^[d]^[k]) :=
  ∑ i, (- IndexType.maxD (fun j => -‖points[i] - centroids[j]‖₂²) 0)


def direction {n k d : ℕ} (points : Float^[d]^[n]) (centroids : Float^[d]^[k]) : Float^[d]^[k] :=
  (let' ((_a,J),(_b,Hdiag)) :=
    ∂> (c:=centroids;VectorType.const 1),
      let' (y,df) := <∂ (kmeansObjective points) c
      (y,df 1)
  VectorType.div J Hdiag)
rewrite_by
  unfold kmeansObjective
  lsimp -zeta (disch:=unsafeAD) only [simp_core,↓revFDeriv_simproc,↓fwdFDeriv_simproc]

def objective (data : KMeansInput) : Float :=
  kmeansObjective data.points data.centroids

def dir (data : KMeansInput) : KMeansOutput :=
  ⟨direction data.points data.centroids⟩
