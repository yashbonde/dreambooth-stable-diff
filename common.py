import os
import time
import json
from PIL import Image
from tqdm import trange
from typing import Dict, List, Any

from datasets import DatasetDict, Dataset


def exponential_backoff(foo, *args, **kwargs):
    max_retries = 5  # Maximum number of retries
    retry_delay = 1  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            out = foo(*args, **kwargs)  # Call the function that may crash
            return out  # If successful, break out of the loop and return
        except Exception as e:
            print(f"Function crashed: {e}")
            if attempt == max_retries - 1:
                print("Max retries reached. Exiting...")
                break
            else:
                delay = retry_delay * (2**attempt)  # Calculate the backoff delay
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait for the calculated delay
    raise ValueError("Exponential back-off failed")


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


def safe_save_files(tracker, *files):
    try:
        exponential_backoff(tracker.save_file, *files)
    except Exception as e:
        print("[ERROR] failed to put files to artifacts:", e)
