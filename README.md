# The stable-diffusion AI startup

In this example we are going to train our own custom model for stable diffusion image generator, monitor it's performance, and serve it as a REST API.


Things

```
alias STD_NBX="nbx projects -id 'a394e541'"
```


### Step 1: 🌤️ Upload Images

First we need to gather some data for this, so create a folder `mkdir data` and put some images in it. Modify the [manifest.jsonl](./manifest.json) file and load with all the images you want to train on with captions. To upload:

```
STD_NBX - artifact put_to manifest.json manifest.json
STD_NBX - artifact put_to ./data ./
```

### Step 2: Train the model

Next to finetune the model we are going to use a GPU NimbleBox Job. To create the job and trigger it.

```
nbx jobs upload src:main \
  --id '<job_id>' \
  --resource_cpu="600m" \
  --resource_memory="600Mi" \
  --resource_disk_size="10Gi" \
  --resource_gpu="nvidia-tesla-k80" \
  --resource_gpu_count="1" \
  --trigger \
  --p "art in the style of @nimblebot"
```

It will create a Relic called "dreambooth" and put all the files there. (Coming) use `nbox.Lmao` to monitor the model in production with a live dashboard.

### Step 3: Serve the model

In order to serve the model for a public end point we are going to use GPU NimbleBox Serving. To create the serving and trigger it.

```
nbx serve upload op_server:prompt \
  --id '<serve_id>' \
  --resource_cpu="600m" \
  --resource_memory="600Mi" \
  --resource_disk_size="10Gi" \
  --resource_gpu="nvidia-tesla-k80" \
  --resource_gpu_count="1" \
  --trigger
```

 

### Finetuning scripts

```
python3 train_text_to_image_lora.py \
  --manifest manifest.json \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --resolution=512 \
  --max_train_steps=4000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=500 \
  --validation_steps 50 \
  --validation_prompt 'Beautiful purple hand holding a red heart with planets, stars and black universe in the background, style: @wrinkledot'
```
