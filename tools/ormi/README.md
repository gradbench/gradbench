# ORMI

[ORMI](https://github.com/l-h-ha/Ormi) is a lazy C++ tensor library with
automatic differentiation and a Python API.

Uses ORMI 0.1.0a8. The Docker image supports Linux x86_64 (`linux/amd64`) only.

To run this tool:

```bash
python3 tools/ormi/run.py
```

## Commentary

`evals.txt` lists enabled evals.

`particle` and `saddle` are disabled. They only implement reverse-over-reverse
differentiation, as forward mode is unsupported in ORMI 0.1.0a8.
