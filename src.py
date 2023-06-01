import json
from tqdm import trange
from typing import Dict, List, Any

from nbox import operator
from nbox.utils import get_files_in_folder
import argparse
import os

# os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/Data/hf_home"

from dreambooth.train_dreambooth import run_training
import torch
from PIL import Image
from huggingface_hub import snapshot_download

from time import time
from nbox import Project

maximum_concepts = 3

"""
v1-5
v2-1-768
v2-1-512
"""


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image


def get_files(dir):
    files = []
    for file in os.listdir(dir):
        files.append(os.path.join(dir, file))
    return files


def load_images(manifest: str, artifact) -> Dict[str, str]:
    with open(manifest, "r") as f:
        data = json.load(f)
    files = [x["path"] for x in data["images"]]
    for _, f in zip(trange(len(files)), files):
        if not os.path.exists(f):
            artifact.get_from(f, f)
    return {x["path"]:x["prompt"] for x in data["images"]}


@operator()
def main(
    manifest: str = "./manifest.json",
    # pretrained_model_name_or_path: str = None,
    # tokenizer_name: str = None,
    # instance_data_dir: str = None,
    # class_data_dir: str = None,
    # instance_prompt: str = None,
    # class_prompt: str = "",
    # with_prior_preservation: bool = False,
    # prior_loss_weight: float = 1.0,
    # num_class_images: int = 100,
    # output_dir: str = "",
    # seed: int = None,
    # resolution: int = 512,
    # center_crop: bool = False,
    # train_text_encoder: bool = False,
    # train_batch_size: int = 4,
    # sample_batch_size: int = 4,
    # gradient_accumulation_steps: int = 1,
    # gradient_checkpointing: bool = False,
    # learning_rate: float = 5e-06,
    # scale_lr: bool = False,
    # lr_scheduler: str = "constant",
    # lr_warmup_steps: int = 500,
    # use_8bit_adam: bool = False,
    # adam_beta1: float = 0.9,
    # adam_beta2: float = 0.999,
    # adam_weight_decay: float = 0.01,
    # adam_epsilon: float = 1e-08,
    # max_grad_norm: float = 1.0,
    # push_to_hub: bool = False,
    # hub_token: str = None,
    # hub_model_id: str = None,
    # logging_dir: str = "logs",
    # mixed_precision: str = "no",
    # save_n_steps: int = 1,
    # save_starting_step: int = 1,
    # stop_text_encoder_training: int = 1000000,
    # image_captions_filename: bool = False,
    # dump_only_text_encoder: bool = False,
    # train_only_unet: bool = False,
    # cache_latents: bool = False,
    # Session_dir: str = "",
    # local_rank: int = -1,
):
    """train your dreambooth model with NimbleBox Jobs. It pulls the data from a folder in the artifacts and trains the model.

    Args:
        p (str, optional): The prompt to fine tune to.
        # pretrained_model_name_or_path(str): Path to pretrained model or model identifier from huggingface.co/models.
        # tokenizer_name(str): Pretrained tokenizer name or path if not the same as model_name
        # instance_data_dir(str): A folder containing the training data of instance images.
        # class_data_dir(str): A folder containing the training data of class images.
        # instance_prompt(str): The prompt with identifier specifying the instance
        # class_prompt(str): The prompt to specify images in the same class as provided
        # with_prior_preservation(bool): Flag to add prior preservation loss.
        # prior_loss_weight(float): The weight of prior preservation loss.
        # num_class_images(int): Minimal class images for prior preservation loss. If not have enough images, additional images will be sampled with class_prompt.
        # output_dir(str): The output directory where the model predictions and checkpoints will be written.
        # seed(int): A seed for reproducible training.
        # resolution(int): The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
        # center_crop(bool): Whether to center crop images before resizing to resolution
        # train_text_encoder(bool): Whether to train the text encoder
        # train_batch_size(int): Batch size (per device) for the training dataloader.
        # sample_batch_size(int): Batch size (per device) for sampling images.
        # gradient_accumulation_steps(int): Number of updates steps to accumulate before performing a backward/update pass.
        # gradient_checkpointing(bool): Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
        # learning_rate(float): Initial learning rate (after the potential warmup period) to use.
        # scale_lr(bool): Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
        # lr_scheduler(str): The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
        # lr_warmup_steps(int): Number of steps for the warmup in the lr scheduler.
        # use_8bit_adam(bool): Whether or not to use 8-bit Adam from bitsandbytes.
        # adam_beta1(float): The beta1 parameter for the Adam optimizer.
        # adam_beta2(float): The beta2 parameter for the Adam optimizer.
        # adam_weight_decay(float): Weight decay to use.
        # adam_epsilon(float): Epsilon value for the Adam optimizer
        # max_grad_norm(float): Max gradient norm.
        # push_to_hub(bool): Whether or not to push the model to the Hub.
        # hub_token(str): The token to use to push to the Model Hub.
        # hub_model_id(str): The name of the repository to keep in sync with the local `output_dir`.
        # logging_dir(str): [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
        # mixed_precision(str): Whether to use mixed precision ["no", "fp16", "bf16"]. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.
        # save_n_steps(int): Save the model every n global_steps
        # save_starting_step(int): The step from which it starts saving intermediary checkpoints
        # stop_text_encoder_training(int): The step at which the text_encoder is no longer trained
        # image_captions_filename(bool): Get captions from filename
        # dump_only_text_encoder(bool): Dump only text encoder
        # train_only_unet(bool): Train only the unet
        # cache_latents(bool): Train only the unet
        # Session_dir(str): Current session directory
        # local_rank(int): For distributed training: local_rank
    """
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != local_rank:
        local_rank = env_local_rank

    # define things that are user everywhere
    which_model = "v1-5"
    resolution = 512  # if which_model != "v2-1-768" else 768
    os.makedirs("instance_images", exist_ok=True)
    os.makedirs("output_model", exist_ok=True)
    torch.cuda.empty_cache()

    project = Project("a394e541")
    artifact = project.get_artifact()
    if not os.path.exists(manifest):
        print("Did not find manifest, downloading manifest")
        artifact.get_from(manifest, manifest)

    # first step is to get all the data and process the images
    print("Loading images ...")
    image_prompts_map = load_images(manifest, artifact)

    file_counter = 0
    for j, (path, prompt) in enumerate(image_prompts_map.items()):
        file = Image.open(path)
        image = pad_image(file)
        image = image.resize((resolution, resolution))
        image = image.convert("RGB")
        image.save(f"instance_images/{prompt}_({j+1}).jpg", format="JPEG", quality=100)
        file_counter += 1
    
    # training variables
    train_text_encoder_for = 15  # 30 for object, 70 for person, 15 for style
    training_Steps = file_counter * 150
    stptxt = int((training_Steps * train_text_encoder_for) / 100)

    # now download the model
    model_v1_5 = snapshot_download(repo_id="multimodalart/sd-fine-tunable")
    model_to_load = model_v1_5
    gradient_checkpointing = True if (which_model != "v1-5") else False
    cache_latents = True if which_model != "v1-5" else False

    args_general = argparse.Namespace(
        image_captions_filename=True,
        train_text_encoder=True if stptxt > 0 else False,
        stop_text_encoder_training=stptxt,
        save_n_steps=0,
        pretrained_model_name_or_path=model_to_load,
        instance_data_dir="instance_images",
        class_data_dir=None,
        output_dir="output_model",
        instance_prompt="",
        seed=42,
        resolution=resolution,
        mixed_precision="fp16",
        train_batch_size=1,
        gradient_accumulation_steps=1,
        use_8bit_adam=True,
        learning_rate=2e-6,
        lr_scheduler="polynomial",
        lr_warmup_steps=0,
        max_train_steps=training_Steps,
        gradient_checkpointing=gradient_checkpointing,
        cache_latents=cache_latents,
    )
    print("Starting single training...")

    run_training(args_imported = args_general, manifest_path = manifest, project = project)
    print("DONE")
 
    print("uploading to artifacts ...")
    artifact.put_to("./output_model/model.ckpt", f"output/{time()}_model.ckpt")

    # it will automatically switch the user agent
    # artifact.set_user_agent(UserAgentType.CURL)
    for i, file in enumerate(get_files_in_folder("./output_model")):
        if "/logs/" in file:
            continue
        op_file = file.replace("/job/output_model/", "")
        op_file = f"output/{op_file}" # for now keep only 1 copy
        print(file, op_file)
        artifact.put_to(file, op_file)

    for i, file in enumerate(get_files_in_folder("./instance_images")):
        op_file = file.replace("/job/instance_images/", "")
        op_file = f"instance_images/{int(time())}_{i}"
        print(file, op_file)
        artifact.put_to(file, op_file)

    


if __name__ == "__main__":
    main()
