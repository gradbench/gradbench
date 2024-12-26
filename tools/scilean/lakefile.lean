import Lake
open Lake DSL

package «gradbench»

@[default_target]
lean_exe «gradbench» where
  root := `Main

lean_lib Gradbench where
  roots := #[`Gradbench]

@[default_target]
lean_exe buildscilean where
  root := `BuildSciLean

require scilean from git "https://github.com/lecopivo/SciLean" @ "master"
