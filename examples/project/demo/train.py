import argparse
import json
import os
from pathlib import Path

from demo.util import square

parser = argparse.ArgumentParser()
parser.add_argument("--value", type=int, default=7)
args = parser.parse_args()
config = json.loads(Path("settings.json").read_text())
result = dict(value=square(args.value), label=config["label"])
if "KGR_INPUT_DATA" in os.environ:
    result["input"] = (Path(os.environ["KGR_INPUT_DATA"]) / "sample.txt").read_text().strip()
output = Path(os.environ["KGR_OUTPUT_DIR"]) / "result.json"
output.write_text(json.dumps(result))
print(json.dumps(result), flush=True)
