nix build '.#eval-norf' '.#eval-qux' '.#tool-bar' '.#tool-baz' '.#tool-foo'
gradbench run --timeout 42 --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-bar' --"
gradbench run --timeout 42 --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-baz' --"
gradbench run --timeout 42 --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-foo' --"
gradbench run --timeout 42 --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-bar' --"
gradbench run --timeout 42 --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-baz' --"
gradbench run --timeout 42 --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-foo' --"
