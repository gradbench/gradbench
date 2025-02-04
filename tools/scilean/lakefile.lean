import Lake
open System Lake DSL

def  moreLinkArgs := #["-lm", "-lblas"]
def  moreLeancArgs : Array String := #[]

package «gradbench» {
  moreLinkArgs := moreLinkArgs
  moreLeancArgs := moreLeancArgs
}

@[default_target]
lean_exe «gradbench» where
  root := `Main
  moreLinkArgs := moreLinkArgs
  moreLeancArgs := moreLeancArgs

lean_lib Gradbench where
  roots := #[`Gradbench]
  moreLinkArgs := moreLinkArgs
  moreLeancArgs := moreLeancArgs

@[default_target]
lean_exe buildscilean where
  root := `BuildSciLean
  moreLinkArgs := moreLinkArgs
  moreLeancArgs := moreLeancArgs

require scilean from git "https://github.com/lecopivo/SciLean" @ "blas"
