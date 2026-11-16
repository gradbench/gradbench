# ORMI

[ORMI](https://github.com/l-h-ha/Ormi) is a lazy C++ tensor library with
automatic differentiation and a Python API.

Uses ORMI 0.1.0a8. The Docker image supports Linux x86_64 (`linux/amd64`) only.

To run this tool:

```bash
python3 tools/ormi/run.py
```

## Commentary

`evals.txt` lists enabled evals. All passed the default inputs. Other modules
are unfinished. Large determinant, LSTM, ODE and particle cases can time out or
exceed memory limits.

`particle` and `saddle` use reverse-over-reverse differentiation. Forward mode
is unsupported in ORMI 0.1.0a8.
