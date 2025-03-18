# OCaml

OCaml is a functional language with an algebraic effect system that
can be used to implement automatic differentiation. The code here was
originally written by [Jesse Sigal][] and integrated into ADBench as
part of his research. The [original code][] is still available. The
parts that are specific to ADBench integration have been removed and
replaced with GradBench-specific code.

[Jesse Sigal]: https://www.jsigal.com/
[original code]: https://github.com/jasigal/ADBench/tree/b98752f96a3b785e07ff6991853dc1073e6bf075/src/ocaml

## Running outside of Docker

You must have [opam][] and [Dune][]. Make sure the `opam` environment
is set - this may require you to run `eval $(opam env)`, although this
is usually done in your shell configuration, and it is not necessary
when using the [shell.nix](../../shell.nix). You may need to do `opam
init` if you have not done so before.

Standing in the root of the GradBench repository, first run

```
$ opam install tools/ocaml --deps-only -y
```

which will install the necessary dependencies. You only need to do
this once.

Then run

```
$ dune build --root tools/ocaml --profile release
```

to actually build the tool. You need to do this whenever you change
the tool. Finally, to run the tool, do

```
$ tools/ocaml/_build/install/default/bin/gradbench
```

although unless you enjoy talking JSON to your computer, you will
actually want to pass this command as the `--tool` argument to
`gradbench`.

[opam]: https://opam.ocaml.org/
[Dune](https://dune.build/)
