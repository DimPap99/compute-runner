import json
import os
from pathlib import Path

out = Path(os.environ["KGR_OUTPUT_DIR"])
print("Hello from Kaggle!", flush=True)
(out / "hello.json").write_text(json.dumps({"ok": True, "job": os.environ["KGR_JOB_ID"]}))
