nix build '.#eval-norf' '.#eval-qux' '.#tool-bar' '.#tool-baz' '.#tool-foo'
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-bar' --"
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-baz' --"
gradbench run --eval "nix run '.#eval-norf' --" --tool "nix run '.#tool-foo' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-bar' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-baz' --"
gradbench run --eval "nix run '.#eval-qux' --" --tool "nix run '.#tool-foo' --"
