"""Bounded GPU and internet smoke check; no training or additional packages."""
import json
import os
import subprocess
import urllib.request
from pathlib import Path

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=30, check=True,
)
assert result.stdout.strip(), "GPU requested but no NVIDIA devices were reported"
with urllib.request.urlopen("https://pypi.org/pypi/pip/json", timeout=20) as response:
    assert response.status == 200
report = {"gpus": result.stdout.strip().splitlines(), "internet": True}
(Path(os.environ["KGR_OUTPUT_DIR"]) / "gpu.json").write_text(json.dumps(report))
print(json.dumps(report), flush=True)
