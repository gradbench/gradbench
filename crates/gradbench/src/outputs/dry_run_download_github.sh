gh run download 15035419296 --name eval-norf --name eval-qux --name tool-bar --name tool-baz --name tool-foo
docker load --input eval-norf/eval-norf.tar
docker load --input eval-qux/eval-qux.tar
docker load --input tool-bar/tool-bar.tar
docker load --input tool-baz/tool-baz.tar
docker load --input tool-foo/tool-foo.tar
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-bar:latest'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-baz:latest'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-norf:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-foo:latest'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-bar:latest'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-baz:latest'
gradbench run --eval 'docker run --rm --interactive ghcr.io/gradbench/eval-qux:latest' --tool 'docker run --rm --interactive ghcr.io/gradbench/tool-foo:latest'
