import json
import shutil
import subprocess
from pathlib import Path

# This script updates `lakefile.lean` and `lean-toolchain` to match the ones in `tools/scilean`

# Load the JSON file
with open('../../tools/scilean/lake-manifest.json', 'r') as f:
    manifest = json.load(f)

# Find the revision for the 'mathlib' package
mathlib_rev = None
for package in manifest['packages']:
    if package['name'] == 'mathlib':
        mathlib_rev = package['rev']
        break

if mathlib_rev is None:
    raise ValueError("Package 'mathlib' not found in lake-manifest.json")

# Generate lakefile.lean
lakefile_content = f"""\
import Lake
open Lake DSL

package «gradbench»

require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "{mathlib_rev}"
"""

# Save to lakefile.lean
with open('lakefile.lean', 'w') as f:
    f.write(lakefile_content)

print(f"Generated lakefile.lean with mathlib revision {mathlib_rev}")

source = Path('../../tools/scilean/lean-toolchain')
destination = Path('lean-toolchain')

if not source.exists():
    raise FileNotFoundError(f"Source lean-toolchain not found at {source}")

shutil.copy(source, destination)
print(f"Copied lean-toolchain to {destination}")


# # Run `lake update`
# try:
#     result = subprocess.run(['lake', 'update'], check=True, text=True, capture_output=True)
#     print("lake update completed successfully:")
#     print(result.stdout)
# except subprocess.CalledProcessError as e:
#     print("Error running lake update:")
#     print(e.stderr)
#     raise
