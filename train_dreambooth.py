import os
import gc
import sys
import math
import json
import random
import argparse
import itertools
import subprocess
from PIL import Image
from pathlib import Path
from pprint import pformat
from tqdm.auto import tqdm
from typing import Optional

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami, snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer

from common import load_images
from nbox import Project # nbox is out SDK / CLI / APIs / python package for NBX


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to training manifest file",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images.")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

    parser.add_argument(
        "--save_n_steps",
        type=int,
        default=1,
        help=("Save the model every n global_steps"),
    )

    parser.add_argument(
        "--save_starting_step",
        type=int,
        default=1,
        help=("The step from which it starts saving intermediary checkpoints"),
    )

    parser.add_argument(
        "--stop_text_encoder_training",
        type=int,
        default=1000000,
        help=("The step at which the text_encoder is no longer trained"),
    )

    parser.add_argument(
        "--image_captions_filename",
        action="store_true",
        help="Get captions from filename",
    )

    parser.add_argument(
        "--dump_only_text_encoder",
        action="store_true",
        default=False,
        help="Dump only text encoder",
    )

    parser.add_argument(
        "--train_only_unet",
        action="store_true",
        default=False,
        help="Train only the unet",
    )

    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Train only the unet",
    )

    parser.add_argument(
        "--Session_dir",
        type=str,
        default="",
        help="Current session directory",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args, unknown = parser.parse_known_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.instance_data_dir is None:
    #    raise ValueError("You must specify a train data directory.")

    # if args.with_prior_preservation:
    #    if args.class_data_dir is None:
    #        raise ValueError("You must specify a data directory for class images.")
    #    if args.class_prompt is None:
    #        raise ValueError("You must specify prompt for class images.")

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        manifest: str,
        tokenizer,
        args,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.image_captions_filename = None

        with open(manifest, "r") as f:
            prompts_data = json.load(f).get("images", [])

        for item in prompts_data:
            path = item["path"]
            if not os.path.exists(path):
                raise ValueError(f"Path: '{path}' doesn't exist.")
    
        self.prompts_data = prompts_data
        self.num_instance_images = len(prompts_data)
        self._length = self.num_instance_images

        if args.image_captions_filename:
            self.image_captions_filename = True

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            random.shuffle(self.class_images_path)
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        data = self.prompts_data[index % self.num_instance_images]
        path = data["path"]
        instance_prompt = data["prompt"]
        instance_image = Image.open(path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def merge_two_dicts(starting_dict: dict, updater_dict: dict) -> dict:
    """
    Starts from base starting dict and then adds the remaining key values from updater replacing the values from
    the first starting/base dict with the second updater dict.

    For later: how does d = {**d1, **d2} replace collision?

    :param starting_dict:
    :param updater_dict:
    :return:
    """
    new_dict: dict = starting_dict.copy()  # start with keys and values of starting_dict
    new_dict.update(updater_dict)  # modifies starting_dict with keys and values of updater_dict
    return new_dict


def merge_args(args1: argparse.Namespace, args2: argparse.Namespace) -> argparse.Namespace:
    """

    ref: https://stackoverflow.com/questions/56136549/how-can-i-merge-two-argparse-namespaces-in-python-2-x
    :param args1:
    :param args2:
    :return:
    """
    # - the merged args
    # The vars() function returns the __dict__ attribute to values of the given object e.g {field:value}.
    merged_key_values_for_namespace: dict = merge_two_dicts(vars(args1), vars(args2))
    args = argparse.Namespace(**merged_key_values_for_namespace)
    return args


def run_training(args, manifest_path: str, project: Project = None):
    logging_dir = Path(args.output_dir, args.logging_dir)
    i = args.save_starting_step
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=torch_dtype
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                with torch.autocast("cuda"):
                    images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    if args.train_only_unet:
        if os.path.exists(str(args.output_dir + "/text_encoder_trained")):
            text_encoder = CLIPTextModel.from_pretrained(args.output_dir, subfolder="text_encoder_trained")
        elif os.path.exists(str(args.output_dir + "/text_encoder")):
            text_encoder = CLIPTextModel.from_pretrained(args.output_dir, subfolder="text_encoder")
        else:
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    else:
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        manifest = manifest_path,
        tokenizer=tokenizer,
        args=args,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.cache_latents:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(text_encoder(batch["input_ids"])[0])
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True
        )

        del vae
        # if not args.train_text_encoder:
        #    del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    def bar(prg):
        br = "|" + "█" * prg + " " * (25 - prg) + "|"
        return br

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    tracker = None
    if project is not None:
        # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
        args_dict = vars(args)
        args_dict["number_of_samples"] = len(train_dataset)
        args_dict["number_batches_each_epoch"] = len(train_dataloader)
        args_dict["total_train_batch_size"] = total_batch_size
        tracker = project.get_exp_tracker(metadata = args_dict)

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if args.cache_latents:
                        latents_dist = batch[0][0]
                    else:
                        latents_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                    latents = latents_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if args.cache_latents:
                    if args.train_text_encoder:
                        encoder_hidden_states = text_encoder(batch[0][1])[0]
                    else:
                        encoder_hidden_states = batch[0][1]
                else:
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            fll = round((global_step * 100) / args.max_train_steps)
            fll = round(fll / 4)
            pr = bar(fll)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            progress_bar.set_description_str("Progress:" + pr)
            accelerator.log(logs, step=global_step)
            if tracker is not None:
                logs["step"] = global_step
                tracker.log(logs)

            if global_step >= args.max_train_steps:
                break

            if args.train_text_encoder and global_step == args.stop_text_encoder_training and global_step >= 30:
                if accelerator.is_main_process:
                    print(" \033[32m" + " Freezing the text_encoder ..." + " \033[0m")
                    frz_dir = args.output_dir + "/text_encoder_frozen"
                    if os.path.exists(frz_dir):
                        subprocess.call("rm -r " + frz_dir, shell=True)
                    os.mkdir(frz_dir)
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                    )
                    pipeline.text_encoder.save_pretrained(frz_dir)

            if args.save_n_steps >= 200:
                if global_step < args.max_train_steps and global_step + 1 == i:
                    ckpt_name = "_step_" + str(global_step + 1)
                    save_dir = Path(args.output_dir + ckpt_name)
                    save_dir = str(save_dir)
                    save_dir = save_dir.replace(" ", "_")
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    inst = save_dir[16:]
                    inst = inst.replace(" ", "_")
                    print(" \033[1;32mSAVING CHECKPOINT: " + args.Session_dir + "/" + inst + ".ckpt")
                    # Create the pipeline using the trained modules and save it.
                    if accelerator.is_main_process:
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                        )
                        pipeline.save_pretrained(save_dir)
                        frz_dir = args.output_dir + "/text_encoder_frozen"
                        if args.train_text_encoder and os.path.exists(frz_dir):
                            subprocess.call("rm -r " + save_dir + "/text_encoder/*.*", shell=True)
                            subprocess.call("cp -f " + frz_dir + "/*.* " + save_dir + "/text_encoder", shell=True)
                        chkpth = args.Session_dir + "/" + inst + ".ckpt"
                        subprocess.call(
                            "python /content/diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py --model_path "
                            + save_dir
                            + " --checkpoint_path "
                            + chkpth
                            + " --half",
                            shell=True,
                        )
                        subprocess.call("rm -r " + save_dir, shell=True)
                        i = i + args.save_n_steps

                    # TODO: @yashbonde complete this part of the code
                    if tracker is not None:
                        if "/logs/" in file:
                            continue
                        tracker.save_file(save_dir)

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        if args.dump_only_text_encoder:
            txt_dir = args.output_dir + "/text_encoder_trained"
            if not os.path.exists(txt_dir):
                os.mkdir(txt_dir)
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
            )
            pipeline.text_encoder.save_pretrained(txt_dir)

        elif args.train_only_unet:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
            )
            pipeline.save_pretrained(args.output_dir)
            txt_dir = args.output_dir + "/text_encoder_trained"
            subprocess.call("rm -r " + txt_dir, shell=True)

        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
            )
            frz_dir = args.output_dir + "/text_encoder_frozen"
            pipeline.save_pretrained(args.output_dir)
            if args.train_text_encoder and os.path.exists(frz_dir):
                subprocess.call("mv -f " + frz_dir + "/*.* " + args.output_dir + "/text_encoder", shell=True)
                subprocess.call("rm -r " + frz_dir, shell=True)

            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

    tracker.end() # end the logging for the tracker


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


def main():
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != local_rank:
        local_rank = env_local_rank
    args_default = parse_args()

    # define things that are user everywhere
    which_model = "v1-5"
    resolution = 512  # if which_model != "v2-1-768" else 768
    os.makedirs("instance_images", exist_ok=True)
    os.makedirs("output_model", exist_ok=True)
    torch.cuda.empty_cache()

    project = Project("a394e541")
    artifact = project.get_artifact()
    manifest_fp = args_default.manifest
    if not os.path.exists(manifest_fp):
        print("Did not find manifest, downloading manifest")
        artifact.get_from(manifest_fp, manifest_fp)

    # first step is to get all the data and process the images
    print("Loading images ...")
    image_prompts_map = load_images(manifest_fp, artifact)

    file_counter = 0
    for j, (path, prompt) in enumerate(image_prompts_map.items()):
        _fp = f"instance_images/{prompt}_({j+1}).jpg"
        if not os.path.exists(_fp):
            file = Image.open(path)
            image = pad_image(file)
            image = image.resize((resolution, resolution))
            image = image.convert("RGB")
            image.save(_fp, format="JPEG", quality=100)
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

    # load the default arguments
    args_imported = argparse.Namespace(
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

    args = merge_args(args_default, args_imported)
    print(pformat(vars(args)))

    # run the training loop and create checkpoint files
    run_training(args = args, manifest_path = manifest_fp, project = project)

    # it will automatically switch the user agent
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
