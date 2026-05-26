gh run download 15035419296 --name eval-norf --name eval-qux --name tool-bar --name tool-baz --name tool-foo
nix-store --import < eval-norf/eval-norf.closure
nix-store --import < eval-qux/eval-qux.closure
nix-store --import < tool-bar/tool-bar.closure
nix-store --import < tool-baz/tool-baz.closure
nix-store --import < tool-foo/tool-foo.closure
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-bar' --"
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-baz' --"
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-foo' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-bar' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-baz' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-foo' --"
