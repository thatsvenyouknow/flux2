"""
Demo client for the FLUX.2 FastAPI server.

Submits a generation request, polls until complete, and saves/displays the image.

Usage:
    python fastapi_deploy/client_demo.py "a cute cat sitting on a windowsill"
    python fastapi_deploy/client_demo.py "a mountain landscape at sunset" --width 1024 --height 512
"""

import argparse
import base64
import sys
import time

import requests

API_URL = "http://localhost:8080"
API_KEY = "dummy"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
POLL_INTERVAL = 2  # seconds between polls


def submit_job(prompt: str, width: int, height: int, seed: int | None) -> str:
    """Submit a generation job and return the job ID."""
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(f"{API_URL}/generate", json=payload, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    print(f"Job submitted: {data['job_id']} (status: {data['status']})")
    return data["job_id"]


def poll_result(job_id: str) -> dict:
    """Poll until the job completes or fails. Returns the result dict."""
    while True:
        resp = requests.get(f"{API_URL}/result/{job_id}", headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()

        status = data["status"]
        if status == "completed":
            print(f"Done! Image: {data['width']}x{data['height']}, seed: {data['seed']}")
            return data
        elif status == "failed":
            print(f"Failed: {data['error']}")
            sys.exit(1)
        else:
            print(f"  Status: {status} ...")
            time.sleep(POLL_INTERVAL)


def save_image(image_b64: str, output_path: str):
    """Decode base64 image and save to disk."""
    img_bytes = base64.b64decode(image_b64)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FLUX.2 API demo client")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--output", "-o", default="generated.png", help="Output file path (default: generated.png)")
    args = parser.parse_args()

    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")

    job_id = submit_job(args.prompt, args.width, args.height, args.seed)
    result = poll_result(job_id)
    save_image(result["image"], args.output)


if __name__ == "__main__":
    main()
