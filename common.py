import os
import json
from PIL import Image
from tqdm import trange
from typing import Dict, List, Any

from datasets import DatasetDict, Dataset

def load_images(manifest: str, artifact) -> Dict[str, str]:
    with open(manifest, "r") as f:
        data = json.load(f)
    files = [x["path"] for x in data["images"]]
    for _, f in zip(trange(len(files)), files):
        if not os.path.exists(f):
            artifact.get_from(f, f)
    return {x["path"]:x["prompt"] for x in data["images"]}

def manifest_to_hf_dataset(manifest_fp: str, artifact) -> DatasetDict:
    with open(manifest_fp, "r") as f:
        data = json.load(f)["images"]
    
    _data = {
        "image": [],
        "prompt": [],
    }
    for _, x in zip(trange(len(data)), data):
        p = x["path"]
        if not os.path.exists(p):
            artifact.get_from(p, p)
        _data["image"].append(Image.open(p))
        _data["prompt"].append(x["prompt"])
    hfdata = DatasetDict({
        "train": Dataset.from_dict(_data)
    })
    return hfdata
