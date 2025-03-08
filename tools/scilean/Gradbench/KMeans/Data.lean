import SciLean

open SciLean Scalar Lean ToJson FromJson

namespace Gradbench.KMeans

set_default_scalar Float

structure KMeansInputRaw where
  points  : Array (Array Float)
  centroids : Array (Array Float)
deriving ToJson, FromJson

structure KMeansInput where
  {d n k : ℕ}
  points : Float^[d]^[n]
  centroids : Float^[d]^[k]

open IndexType in
def KMeansInputRaw.toKMeansInput (data : KMeansInputRaw) : KMeansInput :=
  let d := data.points[0]!.size
  let n := data.points.size
  let k := data.centroids.size
  {
    d := d
    n := n
    k := k
    points := ⊞ (i : Fin n) => ⊞ (j : Fin d) =>
      (data.points.get! i)[j]!
    centroids := ⊞ (i : Fin k) => ⊞ (j : Fin d) =>
      (data.centroids.get! i)[j]!
  }

instance : FromJson KMeansInput where
  fromJson? json := do
    let data : KMeansInputRaw ← fromJson? json
    return data.toKMeansInput

structure KMeansOutput where
  {d k : ℕ}
  output : Float^[d]^[k]

open IndexType in
def KMeansOutput.toArray (data : KMeansOutput) : Array (Array Float) :=
  Array.ofFn (fun (i : Fin data.k) =>
    Array.ofFn (fun (j : Fin data.d) =>
      data.output[i,j]))

instance : ToJson (KMeansOutput) where
  toJson x := toJson x.toArray
