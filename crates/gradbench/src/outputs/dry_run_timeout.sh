docker build . --file evals/norf/Dockerfile --tag ghcr.io/gradbench/eval-norf
docker build . --file evals/qux/Dockerfile --tag ghcr.io/gradbench/eval-qux
docker build . --file tools/bar/Dockerfile --tag ghcr.io/gradbench/tool-bar
docker build . --file tools/baz/Dockerfile --tag ghcr.io/gradbench/tool-baz
docker build . --file tools/foo/Dockerfile --tag ghcr.io/gradbench/tool-foo
gradbench run --timeout 42 --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-bar:latest'
gradbench run --timeout 42 --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-baz:latest'
gradbench run --timeout 42 --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-foo:latest'
gradbench run --timeout 42 --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-bar:latest'
gradbench run --timeout 42 --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-baz:latest'
gradbench run --timeout 42 --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-foo:latest'
