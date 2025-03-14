import SciLean
import Gradbench.KMeans.Data
import Gradbench.Util

namespace Gradbench.KMeans

open SciLean Scalar

set_default_scalar Float

def kmeansObjective {n k d : ℕ} (points : Float^[d]^[n]) (centroids : Float^[d]^[k]) :=
  ∑ᴵ (i : Idx n), minᴵ (j : Idx k), ∑ᴵ (l : Idx d), (points[i,l] - centroids[j,l])^2


def direction {n k d : ℕ} [NeZero k] (points : Float^[d]^[n]) (centroids : Float^[d]^[k]) : Float^[d]^[k] :=
  (let' ((_a,J),(_b,Hdiag)) :=
    ∂> (c:=centroids;VectorType.const 1),
      let' (y,df) := <∂ (kmeansObjective points) c
      (y,df 1)
  VectorType.div J Hdiag)
rewrite_by
  unfold kmeansObjective
  lsimp -zeta (disch:=unsafeAD) only [simp_core,↓revFDeriv_simproc,↓fwdFDeriv_simproc]

def cost (data : KMeansInput) : Float :=
  kmeansObjective data.points data.centroids

def dir (data : KMeansInput) : KMeansOutput :=
  have : NeZero data.k := sorry_proof
  ⟨direction data.points data.centroids⟩
