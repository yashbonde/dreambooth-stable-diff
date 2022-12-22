# This is a bunch of functions to make your life easy
import os
import fire
from tqdm import trange
from nbox import Relics
from nbox.utils import get_files_in_folder

def upload_data_folder(
  relic_name = "dreambooth",
  data_folder = "data/",
):
  ext = ["jpg", "png", "jpeg"]
  ext += [e.upper() for e in ext]
  files = get_files_in_folder(data_folder, ext)
  print("total files:", len(files))

  relic = Relics(relic_name, create=True)
  print(relic)

  targets = []
  for i in trange(len(files)):
    file = files[i]
    target = f"data/{os.path.split(file)[1].replace(' ', '-')}"
    targets.append(target)
    if not relic.has(target):
      relic.put_to(file, target)

  print("total uploaded:", len(targets))
  with open(f'{data_folder}manifest.txt', "w") as f:
    f.write('\n'.join(targets))
  relic.put_to(f'{data_folder}manifest.txt', 'manifest.txt')

if __name__ == "__main__":
  fire.Fire({
    "upload_data_folder": upload_data_folder,
  })
