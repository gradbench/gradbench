import Lake
open System Lake DSL

def moreLinkArgs :=
  if System.Platform.isWindows then
    #[]
  else if System.Platform.isOSX then
    #["-L/opt/homebrew/opt/openblas/lib", "-lblas"]
  else -- assuming linux
    #["-L/usr/lib/x86_64-linux-gnu/", "-lblas", "-lm"]
def moreLeancArgs :=
  if System.Platform.isWindows then
    #[]
  else if System.Platform.isOSX then
    #["-I/opt/homebrew/opt/openblas/include"]
  else -- assuming linux
    #[]

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
