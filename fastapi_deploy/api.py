"""
FastAPI server for FLUX.2 text-to-image generation.

Run with:
    uvicorn fastapi_deploy.api:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import io
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from fastapi_deploy import job_store
from flux2.pipeline import Flux2Pipeline, GenerationOOMError


# ── Pydantic models ──────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for text-to-image generation."""

    prompt: str = Field(..., description="Text prompt describing the image to generate.")
    height: int = Field(1024, ge=64, le=4096, description="Output image height in pixels.")
    width: int = Field(1024, ge=64, le=4096, description="Output image width in pixels.")
    num_steps: int = Field(4, ge=1, le=50, description="Number of denoising steps.")
    guidance: float = Field(4.0, ge=0.0, le=20.0, description="Classifier-free guidance scale.")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility. None = random.")
    reference_images: list[str] = Field(
        default_factory=list,
        description="Base64-encoded reference images (optional). Used for style/content guidance.",
    )


class JobStatus(str, Enum):
    """Possible states of a generation job."""
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class SubmitResponse(BaseModel):
    """Returned immediately when a job is submitted."""
    job_id: str = Field(..., description="Unique job ID. Poll GET /result/{job_id} for the result.")
    status: JobStatus = Field(..., description="Current job status.")


class ResultResponse(BaseModel):
    """Returned when polling for a job result."""
    job_id: str
    status: JobStatus
    image: Optional[str] = Field(None, description="Base64-encoded PNG image (only when status=completed).")
    width: Optional[int] = Field(None, description="Image width (only when status=completed).")
    height: Optional[int] = Field(None, description="Image height (only when status=completed).")
    seed: Optional[int] = Field(None, description="Seed used (only when status=completed).")
    error: Optional[str] = Field(None, description="Error message (only when status=failed).")


# ── API Keys & Rate Limiting ─────────────────────────────────────────────────

API_KEY = "dummy"
requests_per_minute = 2


class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    def is_rate_limited(self, api_key: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        history = self.requests.get(api_key, [])
        history = [t for t in history if t > minute_ago]
        self.requests[api_key] = history

        if len(history) >= self.requests_per_minute:
            return True

        self.requests[api_key].append(now)
        return False


header_scheme = APIKeyHeader(name="X-API-Key", auto_error=True)
rate_limiter = RateLimiter(requests_per_minute)


def verify_api_key_and_rate_limit(api_key: str = Depends(header_scheme)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if rate_limiter.is_rate_limited(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again in a minute.")
    return api_key

def verify_api_key(api_key: str = Depends(header_scheme)):
    """Verify API key without rate limiting. Used for polling endpoints."""
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
# ── Pipeline lifecycle & background worker ────────────────────────────────────

MODEL_NAME = "flux.2-klein-9b-fp8"
pipeline: Optional[Flux2Pipeline] = None
job_queue: asyncio.Queue = None

# Local dict for pending job kwargs (Python objects like file paths can't go
# into Redis). Entries are removed as soon as the worker picks them up.
_pending_kwargs: dict[str, dict] = {}


async def _gpu_worker():
    """Background worker that processes generation jobs one at a time.

    Pulls job IDs from the queue, runs pipeline.generate() in a thread
    (so the event loop stays responsive), and writes results to Redis.
    """
    while True:
        job_id = await job_queue.get()

        # Retrieve and remove the kwargs from the local dict
        kwargs = _pending_kwargs.pop(job_id, None)
        if kwargs is None:
            job_queue.task_done()
            continue

        cond_paths = kwargs.pop("_cond_paths", None)
        job_store.set_processing(job_id)

        try:
            loop = asyncio.get_event_loop()
            result_image = await loop.run_in_executor(
                None, lambda: pipeline.generate(**kwargs)
            )

            # Encode result to base64
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            job_store.set_completed(
                job_id,
                image_b64=image_b64,
                width=result_image.width,
                height=result_image.height,
                seed=kwargs.get("seed") or -1,
            )
        except GenerationOOMError as e:
            job_store.set_failed(job_id, error=str(e))
        except Exception as e:
            job_store.set_failed(job_id, error=f"Generation failed: {e}")
        finally:
            if cond_paths:
                for p in cond_paths:
                    p.unlink(missing_ok=True)
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline at startup, start the GPU worker, clean up on shutdown."""
    global pipeline, job_queue
    pipeline = Flux2Pipeline(model_name=MODEL_NAME)
    pipeline.warmup()

    job_queue = asyncio.Queue()
    worker_task = asyncio.create_task(_gpu_worker())

    yield

    worker_task.cancel()
    del pipeline


app = FastAPI(title="FLUX.2 API", version="0.1.0", lifespan=lifespan)


# ── Helper: decode base64 images to temp files ───────────────────────────────


def _decode_reference_images(images_b64: list[str]) -> list[Path]:
    """Decode base64-encoded images and save to temporary files.

    The pipeline expects file paths, so we write each decoded image to a
    temporary file and return the list of paths.
    """
    paths = []
    for img_b64 in images_b64:
        img_bytes = base64.b64decode(img_b64)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(img_bytes)
        tmp.close()
        paths.append(Path(tmp.name))
    return paths


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Simple health check. Also verifies Redis connectivity."""
    redis_ok = job_store.ping()
    return {"status": "ok" if redis_ok else "degraded", "redis": redis_ok}


@app.post("/generate", response_model=SubmitResponse, status_code=202)
async def generate(request: GenerateRequest, _key: str = Depends(verify_api_key_and_rate_limit)):
    """Submit a generation job. Returns a job ID immediately.

    Poll GET /result/{job_id} to check status and retrieve the image.
    """
    # Decode reference images (if any) to temp files
    cond_paths = _decode_reference_images(request.reference_images) if request.reference_images else None

    # Build kwargs for pipeline.generate()
    gen_kwargs = dict(
        prompt=request.prompt,
        height=request.height,
        width=request.width,
        num_steps=request.num_steps,
        guidance=request.guidance,
        seed=request.seed,
        cond_images=cond_paths,
        _cond_paths=cond_paths,
    )

    # Create job in Redis and stash kwargs locally
    job_id = job_store.create_job()
    _pending_kwargs[job_id] = gen_kwargs

    # Enqueue for the background worker
    await job_queue.put(job_id)

    return SubmitResponse(job_id=job_id, status=JobStatus.queued)


@app.get("/result/{job_id}", response_model=ResultResponse)
def result(job_id: str, _key: str = Depends(verify_api_key)):
    """Poll for the result of a generation job.

    Returns status=queued/processing while the job is pending,
    status=completed with the image when done, or status=failed with
    an error message.
    """
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (may have expired).")

    return ResultResponse(
        job_id=job_id,
        status=job["status"],
        image=job.get("image"),
        width=job.get("width"),
        height=job.get("height"),
        seed=job.get("seed"),
        error=job.get("error"),
    )
