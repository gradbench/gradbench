# Tapenade

[Tapenade][] is an automatic differentiation tool for the [C][] and [Fortran][] programming languages.

For those evals that correspond to ADBench, the code produced by Tapenade requires manual modification. These modified generated programs have been taken from ADBench. See comments in the specific programs for details about the necessary modifications.

To run this outside Docker, you'll first need to run the following commands from the GradBench repository root to install Tapenade into the expected location and build some other binaries:

```sh
wget https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar
tar -xf tapenade_3.16.tar
make -C tools/tapenade
```

Then, to run the tool itself:

```sh
python3 tools/tapenade/run.py
```

[c]: https://en.wikipedia.org/wiki/C_(programming_language)
[fortran]: https://fortran-lang.org/
[tapenade]: https://tapenade.gitlabpages.inria.fr/userdoc/build/html/index.html
