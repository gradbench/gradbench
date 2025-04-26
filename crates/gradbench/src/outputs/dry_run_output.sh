docker build . --file evals/norf/Dockerfile --tag ghcr.io/gradbench/eval-norf:latest
docker build . --file evals/qux/Dockerfile --tag ghcr.io/gradbench/eval-qux:latest
docker build . --file tools/bar/Dockerfile --tag ghcr.io/gradbench/tool-bar:latest
docker build . --file tools/baz/Dockerfile --tag ghcr.io/gradbench/tool-baz:latest
docker build . --file tools/foo/Dockerfile --tag ghcr.io/gradbench/tool-foo:latest
mkdir -p 'a directory/norf' 'a directory/qux'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-bar:latest' -o 'a directory/norf/bar.jsonl'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-baz:latest' -o 'a directory/norf/baz.jsonl'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-foo:latest' -o 'a directory/norf/foo.jsonl'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-bar:latest' -o 'a directory/qux/bar.jsonl'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-baz:latest' -o 'a directory/qux/baz.jsonl'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-foo:latest' -o 'a directory/qux/foo.jsonl'
