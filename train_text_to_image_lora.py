# coding=utf-8
# Copyright 2023 NimbleBox team. All rights reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

# taken from https://github.com/huggingface/diffusers/tree/main/examples/text_to_image

import os
import math
import fire
import random
import logging
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

# import datasets
# from datasets import load_dataset
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from common import load_images, manifest_to_hf_dataset, safe_save_files
from nbox import Project, logger


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")


def main(
    manifest: str,
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
    revision: str = None,
    dataset_config_name: str = None,
    image_column: str = "image",
    caption_column: str = "prompt",
    validation_prompt: str = None,
    num_validation_images: int = 4,
    validation_steps: int = 50,
    max_train_samples: int = None,
    output_dir: str = "sd-model-finetuned-lora",
    cache_dir: str = None,
    seed: int = None,
    resolution: int = 512,
    center_crop: bool = False,
    random_flip: bool = False,
    train_batch_size: int = 16,
    num_train_epochs: int = 100,
    max_train_steps: int = None,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    learning_rate: float = 1e-4,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 500,
    snr_gamma: float = None,
    use_8bit_adam: bool = False,
    allow_tf32: bool = False,
    dataloader_num_workers: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    push_to_hub: bool = False,
    hub_token: str = None,
    hub_model_id: str = None,
    logging_dir: str = "logs",
    mixed_precision: str = None,
    local_rank: int = -1,
    checkpointing_steps: int = 500,
    checkpoints_total_limit: int = None,
    resume_from_checkpoint: str = None,
    enable_xformers_memory_efficient_attention: bool = False,
    noise_offset: float = 0,
) -> None:
    """
    function for training Stable Diffusion using LoRA

    Args:
        manifest (str): Path to training manifest file.
        pretrained_model_name_or_path (str): Path to pretrained model or model identifier from huggingface.co/models.
        revision (str): Revision of pretrained model identifier from huggingface.co/models.
        dataset_config_name (str): The config of the Dataset, leave as None if there's only one config.
        image_column (str): The column of the dataset containing an image.
        caption_column (str): The column of the dataset containing a caption or a list of captions.
        validation_prompt (str): A prompt that is sampled during training for inference.
        num_validation_images (int): Number of images that should be generated during validation with `validation_prompt`.
        validation_steps (int): Run fine-tuning validation every X steps.
        max_train_samples (int): For debugging purposes or quicker training, truncate the number of training examples.
        output_dir (str): The output directory where the model predictions and checkpoints will be written.
        cache_dir (str): The directory where the downloaded models and datasets will be stored.
        seed (int): A seed for reproducible training.
        resolution (int): The resolution for input images.
        center_crop (bool): Whether to center crop the input images to the resolution.
        random_flip (bool): Whether to randomly flip images horizontally.
        train_batch_size (int): Batch size (per device) for the training dataloader.
        num_train_epochs (int): Number of training epochs.
        max_train_steps (int): Total number of training steps to perform.
        gradient_accumulation_steps (int): Number of updates steps to accumulate before performing a backward/update pass.
        gradient_checkpointing (bool): Whether or not to use gradient checkpointing to save memory.
        learning_rate (float): Initial learning rate to use.
        scale_lr (bool): Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
        lr_scheduler (str): The scheduler type to use.
        lr_warmup_steps (int): Number of steps for the warmup in the lr scheduler.
        snr_gamma (float): SNR weighting gamma to be used if rebalancing the loss.
        use_8bit_adam (bool): Whether or not to use 8-bit Adam from bitsandbytes.
        allow_tf32 (bool): Whether or not to allow TF32 on Ampere GPUs.
        dataloader_num_workers (int): Number of subprocesses to use for data loading.
        adam_beta1 (float): The beta1 parameter for the Adam optimizer.
        adam_beta2 (float): The beta2 parameter for the Adam optimizer.
        adam_weight_decay (float): Weight decay to use.
        adam_epsilon (float): Epsilon value for the Adam optimizer.
        max_grad_norm (float): Max gradient norm.
        push_to_hub (bool): Whether or not to push the model to the Hub.
        hub_token (str): The token to use to push to the Model Hub.
        hub_model_id (str): The name of the repository to keep in sync with the local `output_dir`.
        logging_dir (str): TensorBoard log directory.
        mixed_precision (str): Whether to use mixed precision.
        local_rank (int): For distributed training: local_rank.
        checkpointing_steps (int): Save a checkpoint of the training state every X updates.
        checkpoints_total_limit (int): Max number of checkpoints to store.
        resume_from_checkpoint (str): Whether training should be resumed from a previous checkpoint.
        enable_xformers_memory_efficient_attention (bool): Whether or not to use xformers.
        noise_offset (float): The scale of noise offset.
    """

    args = argparse.Namespace(
        manifest=manifest,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        dataset_config_name=dataset_config_name,
        image_column=image_column,
        caption_column=caption_column,
        validation_prompt=validation_prompt,
        num_validation_images=num_validation_images,
        validation_steps=validation_steps,
        max_train_samples=max_train_samples,
        output_dir=output_dir,
        cache_dir=cache_dir,
        seed=seed,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
        train_batch_size=train_batch_size,
        num_train_epochs=num_train_epochs,
        max_train_steps=max_train_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        scale_lr=scale_lr,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        snr_gamma=snr_gamma,
        use_8bit_adam=use_8bit_adam,
        allow_tf32=allow_tf32,
        dataloader_num_workers=dataloader_num_workers,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_weight_decay=adam_weight_decay,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm,
        push_to_hub=push_to_hub,
        hub_token=hub_token,
        hub_model_id=hub_model_id,
        logging_dir=logging_dir,
        mixed_precision=mixed_precision,
        local_rank=local_rank,
        checkpointing_steps=checkpointing_steps,
        checkpoints_total_limit=checkpoints_total_limit,
        resume_from_checkpoint=resume_from_checkpoint,
        enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
        noise_offset=noise_offset
    )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    project = Project("a394e541")
    artifact = project.get_artifact()
    manifest_fp = args.manifest
    if not os.path.exists(manifest_fp):
        print("Did not find manifest, downloading manifest")
        artifact.get_from(manifest_fp, manifest_fp)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    # freeze parameters of models to save more memoryq
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(lora_attn_procs)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # NOTE: The only advantage of using the hf-datasets is that we can apply transformations on it
    # so we will need to build our thing that returns the DatasetDict thingy. however in the example code
    # that uses the pokemon captions (https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions)
    # dataset, it will also load all the images in the memory, lol!
    # TODO: @yashbonde this is the point that can be improved, by providing a similar streaming layer in the
    # middle

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # # download the dataset.
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #     )
    # else:
    #     data_files = {}
    #     if args.train_data_dir is not None:
    #         data_files["train"] = os.path.join(args.train_data_dir, "**")
    #     dataset = load_dataset(
    #         "imagefolder",
    #         data_files=data_files,
    #         cache_dir=args.cache_dir,
    #     )
    #     # See more about loading custom images at
    #     # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # can concurrency become a big issue?
    print("Loading images ...")
    dataset = manifest_to_hf_dataset(manifest_fp = args.manifest, artifact = artifact)

    # 6. Get the column names for input/target.
    column_names = dataset["train"].column_names
    image_column = args.image_column
    caption_column = args.caption_column

    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples["prompt"]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range())
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
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

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if accelerator.is_main_process:
        # create a NimbleBox tracker
        # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
        args_dict = vars(args)
        args_dict["number_of_samples"] = len(train_dataset)
        args_dict["total_train_batch_size"] = total_batch_size
        tracker = project.get_exp_tracker(metadata = args_dict)


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "train_loss": train_loss, "step": global_step}
            progress_bar.set_postfix(**logs)
            if accelerator.sync_gradients:                
                tracker.log(logs)
                progress_bar.update(1)
                global_step += 1
                # accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # upload the checkpoint to Artifacts
                        safe_save_files(tracker, save_path)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process and args.validation_prompt is not None and global_step % args.validation_steps == 0:
            logger.info(
                f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                f" {args.validation_prompt}."
            )
            # create pipeline
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            images = []
            for _ in range(args.num_validation_images):
                images.append(
                    pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                )

            # store the images in the artifacts
            fps = []
            for i, image in enumerate(images):
                fp = f"{global_step}_{i:02d}_{args.validation_prompt}.png"
                image.save(fp)
                fps.append(fp)
            safe_save_files(tracker, *fps)

            del pipeline
            torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)
        safe_save_files(tracker, args.output_dir)
        

    # Final inference
    # Load previous pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(accelerator.device)

    # load attention processors
    pipeline.unet.load_attn_procs(args.output_dir)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

    if accelerator.is_main_process:
        # store the images in the artifacts
        fps = []
        for i, image in enumerate(images):
            fp = f"{global_step}_{i:02d}_{args.validation_prompt}.png"
            image.save(fp)
            fps.append(fp)
        safe_save_files(tracker, *fps)

    accelerator.end_training()
    tracker.end()

if __name__ == "__main__":
    fire.Fire(main)
