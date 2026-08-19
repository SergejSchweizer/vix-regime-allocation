from pathlib import Path

import nbformat

path = Path("notebooks/gwp2_vix_regime_allocation.ipynb")
nb = nbformat.read(path, as_version=4)
fixed = False
for cell in reversed(nb.cells):
    if cell.cell_type == "code" and "_parse_bibtex_registry" in cell.source:
        if cell.source.startswith("import re\n\n"):
            cell.source = cell.source[len("import re\n\n") :]
            fixed = True
        break
if not fixed:
    raise SystemExit("Expected redundant import re was not found.")
nbformat.validate(nb)
nbformat.write(nb, path)
