nix build --no-link --print-out-paths '.#eval-hello'
nix build --no-link --print-out-paths '.#eval-llsq'
nix build --no-link --print-out-paths '.#tool-jax'
nix build --no-link --print-out-paths '.#tool-manual'
nix build --no-link --print-out-paths '.#tool-pytorch'
gradbench run --eval "nix build --no-link --print-out-paths --offline '.#eval-hello' | xargs -I{} {}/bin/run" --tool "nix build --no-link --print-out-paths --offline '.#tool-jax' | xargs -I{} {}/bin/run"
gradbench run --eval "nix build --no-link --print-out-paths --offline '.#eval-hello' | xargs -I{} {}/bin/run" --tool "nix build --no-link --print-out-paths --offline '.#tool-manual' | xargs -I{} {}/bin/run"
gradbench run --eval "nix build --no-link --print-out-paths --offline '.#eval-hello' | xargs -I{} {}/bin/run" --tool "nix build --no-link --print-out-paths --offline '.#tool-pytorch' | xargs -I{} {}/bin/run"
gradbench run --eval "nix build --no-link --print-out-paths --offline '.#eval-llsq' | xargs -I{} {}/bin/run" --tool "nix build --no-link --print-out-paths --offline '.#tool-jax' | xargs -I{} {}/bin/run"
gradbench run --eval "nix build --no-link --print-out-paths --offline '.#eval-llsq' | xargs -I{} {}/bin/run" --tool "nix build --no-link --print-out-paths --offline '.#tool-manual' | xargs -I{} {}/bin/run"
gradbench run --eval "nix build --no-link --print-out-paths --offline '.#eval-llsq' | xargs -I{} {}/bin/run" --tool "nix build --no-link --print-out-paths --offline '.#tool-pytorch' | xargs -I{} {}/bin/run"
