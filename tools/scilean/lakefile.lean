import Lake
open System Lake DSL

def moreLinkArgs :=
  if System.Platform.isWindows then
    #[]
  else if System.Platform.isOSX then
    #["-L/opt/homebrew/opt/openblas/lib",
      "-L/usr/local/opt/openblas/lib", "-lblas"]
  else -- assuming linux
    #["-L/usr/lib/x86_64-linux-gnu/", "-lblas", "-lm"]

package «gradbench» {
  moreLinkArgs := moreLinkArgs
}

@[default_target]
lean_exe «gradbench» where
  root := `Main

lean_lib Gradbench where
  roots := #[`Gradbench]

@[default_target]
lean_exe buildscilean where
  root := `BuildSciLean

require scilean from git "https://github.com/lecopivo/SciLean" @ "blas"
