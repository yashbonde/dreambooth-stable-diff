import os
import io
import torch
import base64
import uvicorn
from PIL import Image
from time import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from nbox import Project

PROJECT_ID = "a394e541"
TRACKER_ID = "x-00014-Melbourne"


class ImageCreate(BaseModel):
    seed: Optional[int] = 42
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    prompt: str


model_path = os.getenv("MODEL_PATH", f"../bundle-a394e541/{TRACKER_ID}/sd-model-finetuned-lora")
print(f"Load the model from path ... '{model_path}'")

st = time()
project = Project(PROJECT_ID)
if not os.path.exists(model_path):
    # download the model weights from artifacts
    os.path.makedirs("sd-model-finetuned-lora")
    model_path = "sd-model-finetuned-lora"
    artifact = project.get_artifact()
    artifact.get_from(model_path + "/pytorch_lora_weights.bin", f"{TRACKER_ID}/sd-model-finetuned-lora/pytorch_lora_weights.bin")

if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the pipe 
model_base = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.unet.load_attn_procs(model_path)
pipe.to(device)

print(f"... took {time() - st}s to get ready")

app = FastAPI()

@app.get("/")
def read_root():
    return "I'm doin oK"

@app.post("/api/generate/")
async def generate_image(prompt: ImageCreate):
    generator = None if prompt.seed is None else torch.Generator().manual_seed(int(prompt.seed))
    image: Image = pipe(
        prompt.prompt,
        guidance_scale=prompt.guidance_scale, 
        num_inference_steps=min(10, prompt.num_inference_steps), 
        generator = generator, 
    ).images[0]

    # Convert image to base64
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    base64_image = base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    # Create the response dictionary
    response = {
        "image": base64_image,
        "type": "png",
        "resolution": [image.width, image.height]
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
