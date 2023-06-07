import base64
import fire
import requests

def main(
    p: str = "Astronaut in a Jungle by Syd Mead, cold color palette, muted colors, detailed, 8k",
    n: int = 10,
    fp: str = "sample.png",
    seed = 42069,
    guide = 7,
):
    # Send the request to the server
    r = requests.post(
        "http://localhost:8001/api/generate/",
        json={
            "prompt": p,
            "seed": seed,
            "guidance_scale": guide,
            "num_inference_steps": n
        }
    )

    r.raise_for_status()
    data = r.json()

    # Access the image details from the r
    image_base64 = data["image"]
    image_type = data["type"]
    image_resolution = data["resolution"]

    # Convert base64 image data to bytes
    image_bytes = base64.b64decode(image_base64)

    # Save the image to a file
    with open(fp, "wb") as file:
        file.write(image_bytes)

    print("Image saved successfully!")


if __name__ == "__main__":
    fire.Fire(main)