import SciLean
import Gradbench.LLSQ.Data

open SciLean Lean

set_default_scalar Float

namespace Gradbench.LLSQ

def _root_.Float.sign (x : Float) : Float := if 0 < x then 1.0 else if x < 0 then -1.0 else 0.0

def _root_.SciLean.Idx.toFloat {n} (i : Idx n) : Float := i.1.toUInt64.toFloat


-- direct implementation of the formula
def primal_v1 (n : ℕ) (x : Float^[m]) : Float :=
  let t (i : Idx n) : Float := -1 + i.toFloat*2/(n.toFloat-1)
  let s i := (t i).sign
  0.5 * ∑ᴵ i, (s i - ∑ᴵ j, x[j] * (t i)^(j.toFloat))^2

def gradient_v1 (n : ℕ) (x : Float^[m]) : Float^[m] :=
    ((<∂ (x':=x), primal_v1 n x').2 1)
  rewrite_by
    unfold primal_v1
    autodiff


-- Horner scheme
def primal_v2 (n : ℕ) (x : Float^[m]) : Float :=
  let t (i : Idx n) : Float := -1 + i.toFloat*2/(n.toFloat-1)
  0.5 * ∑ᴵ i,
    let ti := t i
    let si := ti.sign
    (si - IndexType.fold IndexType.Range.full.reverse (init:=0.0)
            (fun (i : Idx m) (s : Float) => s*ti + x[i]))^2


def gradient_v2 (n : ℕ) (x : Float^[m]) : Float^[m] :=
    ((<∂ (x':=x), primal_v2 n x').2 1)
  rewrite_by
    unfold primal_v2
    autodiff


-- vectorized
def primal_v3 (n : ℕ) (x : Float^[m]) : Float :=
  let t := ⊞ (i : Idx n) => -1 + i.toFloat*2/(n.toFloat-1)
  let s := t.rmap .sign
  let T := ⊞ i (j : Idx m) => (t[i])^(j:Float)
  ‖s + T*x‖₂²


def gradient_v3 (n : ℕ) (x : Float^[m]) : Float^[m] :=
    ((<∂ (x':=x), primal_v3 n x').2 1)
  rewrite_by
    unfold primal_v3
    lsimp -zeta only [simp_core, revFDeriv_simproc]


def primal (input : LLSQInput) : Float := primal_v1 input.n input.x
def gradient (input : LLSQInput) : DataArray Float := (gradient_v1 input.n input.x).1
