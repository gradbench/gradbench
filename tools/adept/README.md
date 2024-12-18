# Adept

[Adept](https://www.met.reading.ac.uk/clouds/adept/) is a C++ library
that uses operator overloading to implement AD. It mostly focuses on
reverse mode, but is also effective in forward mode.

To run this outside Docker, you'll first need to run the following
command from the GradBench repository root to setup the C++ build and
build the programs:

```sh
make -C cpp
```

You need to download and compile Adept and set the environment
variable `ADEPT_DIR` to point to the directory containing Adept (you
don't need to install it, but you do need to compile it). See the
[Dockerfile](Dockerfile) for details.

Then, to run the tool itself:

```sh
python3 tools/adept/run.py
```
