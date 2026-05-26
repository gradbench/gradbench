nix build '.#eval-norf'
nix build '.#eval-qux'
nix build '.#tool-bar'
nix build '.#tool-baz'
nix build '.#tool-foo'
gradbench run --timeout 42 --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-bar' --"
gradbench run --timeout 42 --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-baz' --"
gradbench run --timeout 42 --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-foo' --"
gradbench run --timeout 42 --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-bar' --"
gradbench run --timeout 42 --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-baz' --"
gradbench run --timeout 42 --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-foo' --"
