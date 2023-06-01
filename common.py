import os
import json
from tqdm import trange
from typing import Dict, List, Any

def load_images(manifest: str, artifact) -> Dict[str, str]:
    with open(manifest, "r") as f:
        data = json.load(f)
    files = [x["path"] for x in data["images"]]
    for _, f in zip(trange(len(files)), files):
        if not os.path.exists(f):
            artifact.get_from(f, f)
    return {x["path"]:x["prompt"] for x in data["images"]}
