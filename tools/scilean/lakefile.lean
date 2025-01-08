import Lake
open System Lake DSL

package «gradbench»

@[default_target]
lean_exe «gradbench» where
  root := `Main
  moreLinkArgs := #["-lm", "-lblas"]

lean_lib Gradbench where
  roots := #[`Gradbench]
  moreLinkArgs := #["-lm", "-lblas"]

@[default_target]
lean_exe buildscilean where
  root := `BuildSciLean
  moreLinkArgs := #["-lm", "-lblas"]

require scilean from git "https://github.com/lecopivo/SciLean" @ "blas"
