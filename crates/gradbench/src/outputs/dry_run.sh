nix build '.#eval-norf'
nix build '.#eval-qux'
nix build '.#tool-bar'
nix build '.#tool-baz'
nix build '.#tool-foo'
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-bar' --"
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-baz' --"
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-foo' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-bar' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-baz' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-foo' --"
