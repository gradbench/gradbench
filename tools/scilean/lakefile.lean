import Lake
open Lake DSL

package «gradbench»

@[default_target]
lean_exe «gradbench» where
  root := `Main
