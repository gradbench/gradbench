import Lake
open Lake DSL

package «gradbench»

@[default_target]
lean_exe «gradbench» where
  root := `Main

require scilean from git "https://github.com/lecopivo/SciLean.git" @ "22d53b2f4e3db2a172e71da6eb9c916e62655744"
