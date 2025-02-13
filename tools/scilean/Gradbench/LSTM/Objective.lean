import SciLean
import Gradbench.LSTM.Data

open SciLean Scalar Lean ToJson FromJson

namespace Gradbench.LSTM

set_option pp.deepTerms true
set_option pp.proofs false
-- set_option profiler true

local macro (priority:=high+1) "Float^[" M:term ", " N:term "]" : term =>
  `(FloatMatrix' .RowMajor .normal (Fin $M) (Fin $N))

local macro (priority:=high+1) "Float^[" N:term "]" : term =>
  `(FloatVector (Fin $N))

set_default_scalar Float

open Scalar in
def sigmoid {R} [RealScalar R] (x : R) : R := 1 / (1 + exp (-x))

abbrev_data_synth sigmoid in x : HasRevFDeriv R by
  unfold sigmoid
  data_synth (disch:=sorry_proof) => enter[3]; simp; to_ssa; to_ssa; lsimp

abbrev_data_synth sigmoid in x : HasRevFDerivUpdate R by
  unfold sigmoid
  data_synth (disch:=sorry_proof) => enter[3]; simp; to_ssa; to_ssa; lsimp


open Scalar in
abbrev_data_synth Scalar.tanh in x : HasRevFDeriv K by
  conv => enter[3]; assign (fun x : K =>
    let y := tanh x
    (y, fun dy => dy*(1 - y^2)))
  sorry_proof


@[simp,simp_core]
theorem VectorType.conj_real (x : Float^[n]) : VectorType.conj x = x := sorry_proof

open VectorType MatrixType in
def lstmModel {d : ℕ}
              (weight: Float^[4,d])
              (bias: Float^[4,d])
              (hidden: Float^[d])
              (cell: Float^[d])
              (input: Float^[d]) : Float^[d] × Float^[d] :=
  let forget  := input  |> (mul · (row weight 0)) |> (· + (row bias 0)) |> (map sigmoid ·)
  let ingate  := hidden |> (mul · (row weight 1)) |> (· + (row bias 1)) |> (map sigmoid ·)
  let outgate := input  |> (mul · (row weight 2)) |> (· + (row bias 2)) |> (map sigmoid ·)
  let change  := hidden |> (mul · (row weight 3)) |> (· + (row bias 3)) |> (map tanh ·)
  let t1s := mul cell forget
  let t2s := mul ingate change
  let cell2 := t1s + t2s
  let hidden2 := mul outgate (map tanh cell2)
  (hidden2, cell2)

set_option maxRecDepth 1000000

def_data_synth lstmModel in weight bias hidden cell input : HasRevFDeriv Float by
  unfold lstmModel VectorType.map; dsimp -zeta
  data_synth => enter[3]; lsimp

def_data_synth lstmModel in weight bias hidden cell input : HasRevFDerivUpdate Float by
  unfold lstmModel VectorType.map; dsimp -zeta
  data_synth => enter[3]; lsimp

open VectorType MatrixType in
def lstmPredict {slen d : ℕ}
                (mainParams: (Float^[4,d])^[slen,2])
                (extraParams: Float^[3,d])
                (state: (Float^[d])^[slen,2])
                (input: Float^[d]) : Float^[d] × Float^[d]^[slen,2] :=
  let x₀ := mul input (row extraParams 0)
  let state₀ : (Float^[d])^[slen,2] := 0

  let' (state',x') := IndexType.foldl (init:=(state₀,x₀))
    (fun sx (i : Fin slen) =>
      let' (s,x) := sx
      let' (h,c) := lstmModel mainParams[i,0] mainParams[i,1] state[i,0] state[i,1] x
      let s := ArrayType.set s (i,0) h
      let s := ArrayType.set s (i,1) c
      (s,h))

  let v' := mul x' (row extraParams 1) + (row extraParams 2)
  (v', state')


def_data_synth lstmPredict in mainParams extraParams state : HasRevFDeriv Float by
  unfold lstmPredict; --dsimp -zeta
  data_synth => enter[3]; lsimp


open VectorType in
abbrev_data_synth Scalar.log in x : HasRevFDeriv K by
  conv => enter[3]; assign (fun x => (Scalar.log x, fun dx : K => dx/x))
  sorry_proof

abbrev_data_synth Scalar.log in x : HasRevFDerivUpdate K by
  conv => enter[3]; assign (fun x => (Scalar.log x, fun (dx : K) dx' => dx' + dx/x))
  sorry_proof


open VectorType MatrixType in
def lstmObjective {slen lenSeq d : ℕ}
                  (mainParams : (Float^[4,d])^[slen,2])
                  (extraParams : Float^[3,d])
                  (state : (Float^[d])^[slen, 2])
                  (sequence : Float^[d]^[lenSeq]) : Float :=
  -- state : [stlen][2][d]f64
  let' (_a, total) := IndexType.foldl (init:=(state, (0:Float)))
    fun st (i : Fin (lenSeq - 1)) =>
      let' (oldState, oldTotal) := st
      let' (y_pred, newState) := lstmPredict mainParams extraParams oldState sequence[⟨i.1,sorry_proof⟩]
      -- y_pred: DV [d]f64, newState: DM
      let tmp_sum := sum (exp y_pred)
      let tmp_log := - Scalar.log (tmp_sum + 2.0)
      let ynorm := scalAdd 1 tmp_log y_pred
      let newTotal := oldTotal + (⟪sequence[⟨i.1 + 1,sorry_proof⟩], ynorm⟫)
      (newState, newTotal)

  let count := d * (lenSeq - 1) |>.toFloat
  (-total * count⁻¹)

open VectorType in
abbrev_data_synth scalAdd in beta x [Lawful X] : HasRevFDeriv K by
  simp only [blas_to_module]
  data_synth => enter[3]; simp[vector_optimize]; to_ssa; to_ssa; lsimp

open VectorType in
abbrev_data_synth scalAdd in beta x [Lawful X] : HasRevFDerivUpdate K by
  simp only [blas_to_module]
  data_synth => enter[3]; simp[vector_optimize]; to_ssa; to_ssa; lsimp


abbrev_data_synth lstmObjective in mainParams extraParams : HasRevFDeriv Float by
  unfold lstmObjective; --dsimp -zeta
  data_synth => enter[3]; lsimp


def objective (input : LSTMInput) : Float :=
  lstmObjective input.mainParams input.extraParams input.state input.sequence


set_option linter.unusedVariables false in
def lstmJacobian {slen lenSeq d : ℕ}
                 (mainParams : (Float^[4,d])^[slen,2])
                 (extraParams : Float^[3,d])
                 (state : (Float^[d])^[slen, 2])
                 (sequence : Float^[d]^[lenSeq])
  {f'} (deriv : HasRevFDeriv Float (fun (x,y) => lstmObjective x y state sequence) f')
  {grad} (simp : grad = (f' (mainParams,extraParams)).2 1) := grad


def jacobian (input : LSTMInput) : LSTMOutput :=
  let (x,y) := lstmJacobian input.mainParams input.extraParams input.state input.sequence
     (deriv := by data_synth)
     (simp := by lsimp; rfl)
  ⟨x,y⟩
