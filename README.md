# The stable-diffusion AI startup

In this example we are going to train our own custom model for stable diffusion image generator, monitor it's performance, and serve it as a REST API.


### Step 1: Create a folder with images

First we need to gather some data for this, so create a folder `mkdir data` and put some images in it.

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

### Step 3: Serve the model (Upcoming)

In order to serve the model for a public end point we are going to use GPU NimbleBox Serving. To create the serving and trigger it.

```
nbx serve upload op_server:prompt \
  --id '<serve_id>' \
  --resource_cpu="600m" \
  --resource_memory="600Mi" \
  --resource_disk_size="10Gi" \
  --resource_gpu="nvidia-tesla-k80" \
  --resource_gpu_count="1" \
  --trigger \
  --p "a boat floating in the sky, art in the style of @nimblebot"
```

 