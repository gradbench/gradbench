import Lake
open System Lake DSL

package «gradbench»

@[default_target]
lean_exe «gradbench» where
  root := `Main

lean_lib Gradbench where
  roots := #[`Gradbench]
  moreLinkArgs := #["-lm"]

@[default_target]
lean_exe buildscilean where
  root := `BuildSciLean

require scilean from git "https://github.com/lecopivo/SciLean" @ "master"


target ffi.o pkg : FilePath := do
  let oFile := pkg.buildDir / "c" / "ffi.o"
  let srcJob ← inputFile (text:=true) <| pkg.dir / "c" / "lgamma.c"
  let weakArgs := #["-I", (← getLeanIncludeDir).toString]
  buildO oFile srcJob weakArgs #["-fPIC"] "gcc"

extern_lib libleanffi pkg := do
  let name := nameToStaticLib "leanffi"
  let ffiO ← fetch <| pkg.target ``ffi.o
  buildStaticLib (pkg.nativeLibDir / name) #[ffiO]
