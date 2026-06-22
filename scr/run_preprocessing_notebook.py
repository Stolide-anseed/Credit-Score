import json
from pathlib import Path


path = Path("../notebooks") / "preprocessing.ipynb"
notebook = json.loads(path.read_text(encoding="utf-8"))

namespace = {"__name__": "__main__"}
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", "")
    if not source.strip():
        continue
    print(f"Executing code cell {index}")
    exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)

print("Notebook code cells executed successfully")
