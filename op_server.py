import os
from functools import lru_cache
from nbox import Relics, operator
from nbox.sublime.proto.relics_rpc_pb2 import ListRelicFilesRequest
from nbox.auth import secret, ConfigString

import torch
from diffusers import DiffusionPipeline


@lru_cache()
def get_model():
  relic = Relics("dreambooth")

  def get_files_in_folder(folder: str = "output/"):
    files = []
    out = relic.stub.list_relic_files(ListRelicFilesRequest(
      prefix = folder,
      relic_name = "dreambooth",
      workspace_id = secret.get(ConfigString.workspace_id)
    ))
    # print(out)
    for f in out.files:
      if f.type == 1:
        sub_files = get_files_in_folder(f.name+"/")
        files.extend(sub_files)
      else:
        files.append(f.name)
    return files

  files_to_download = list(filter(
    lambda x: " " not in x,
    get_files_in_folder(folder = "output/")
  ))
  files_to_download

  for f in files_to_download:
    folders = ["/".join(f.split("/")[:i+1]) for i in range(len(f.split("/"))-1)]
    for _f in folders:
      os.makedirs(_f, exist_ok=True)
    relic.get_from(f, f)

  pipeline = DiffusionPipeline.from_pretrained("output")
  pipeline = pipeline.to("cuda")

  return pipeline

@operator()
def prompt(
  p: str,
  h: int = 512,
  w: int = 512,
  num_inference_steps: int = 50,
  guidance_scale: float = 7.5,
  negative_prompt: str = None
):
  pipeline = get_model()
  with torch.no_grad():
    out = pipeline(p, h, w, num_inference_steps, guidance_scale, negative_prompt)
  return {
    "mime_type": "image/jpeg",
    "b64": out.images[0].convert("RGB").tobytes("jpeg", "RGB"),
    "request": {
      "p": p,
      "h": h,
      "w": w,
      "num_inference_steps": num_inference_steps,
      "guidance_scale": guidance_scale,
      "negative_prompt": negative_prompt
    }
  }

# just call this to load the model and cache objects, it's okay to put this code here
# since we are not going to call until this is on the pod
prompt("hello world")
