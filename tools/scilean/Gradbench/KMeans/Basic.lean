import SciLean

open SciLean

local macro (priority:=high+1) "Float^[" M:term ", " N:term "]" : term =>
  `(FloatMatrix' .RowMajor .normal (Fin $M) (Fin $N))

local macro (priority:=high+1) "Float^[" N:term "]" : term =>
  `(FloatVector (Fin $N))

set_default_scalar Float

macro A:ident "[" i:term "," ":" "]" : term => `(MatrixType.row $A $i)

def Float.inf : Float := 1.0/0.0

def objective (points : Float^[n,d]) (centroids : Float^[k,d]) : Float := Id.run do

  let mut s : Float := 0
  for i in fullRange (Fin n) do

    let xi := points[i,:]
    let mut dist := Float.inf

    for j in fullRange (Fin k) do
      let cj := centroids[j,:]
      let distij := ‖xi - cj‖₂
      if distij < dist then
        dist := distij

    s += s
  return s
