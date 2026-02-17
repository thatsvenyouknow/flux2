"""
Redis-backed job store for generation requests.

Each job is stored as a Redis hash under the key ``job:{job_id}`` with fields:
    status   – queued | processing | completed | failed
    image    – base64-encoded PNG (only when completed)
    width    – image width (only when completed)
    height   – image height (only when completed)
    seed     – seed used (only when completed)
    error    – error message (only when failed)

All keys expire automatically after JOB_TTL_SECONDS.
"""

import os
import uuid

import redis

# How long completed/failed jobs stay in Redis before auto-deletion
JOB_TTL_SECONDS = 600  # 10 minutes

# Connect to Redis using the URL from the environment variable.
# decode_responses=True makes redis return Python strings instead of bytes.
_redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_r = redis.Redis.from_url(_redis_url, decode_responses=True)


def _key(job_id: str) -> str:
    """Build the Redis key for a job. All jobs live under the ``job:`` prefix."""
    return f"job:{job_id}"


def ping() -> bool:
    """Check if Redis is reachable."""
    try:
        return _r.ping()
    except redis.ConnectionError:
        return False


def create_job() -> str:
    """Create a new job in 'queued' state. Returns the job ID."""
    job_id = str(uuid.uuid4())
    _r.hset(_key(job_id), mapping={"status": "queued"})
    _r.expire(_key(job_id), JOB_TTL_SECONDS)
    return job_id


def set_processing(job_id: str) -> None:
    """Mark a job as currently being processed by the GPU worker."""
    _r.hset(_key(job_id), "status", "processing")


def set_completed(job_id: str, image_b64: str, width: int, height: int, seed: int) -> None:
    """Mark a job as completed and store the result."""
    _r.hset(_key(job_id), mapping={
        "status": "completed",
        "image": image_b64,
        "width": str(width),
        "height": str(height),
        "seed": str(seed),
    })
    # Reset TTL so the result stays available for another full window
    _r.expire(_key(job_id), JOB_TTL_SECONDS)


def set_failed(job_id: str, error: str) -> None:
    """Mark a job as failed and store the error message."""
    _r.hset(_key(job_id), mapping={
        "status": "failed",
        "error": error,
    })
    _r.expire(_key(job_id), JOB_TTL_SECONDS)


def get_job(job_id: str) -> dict | None:
    """Retrieve all fields of a job. Returns None if the job doesn't exist (expired or invalid)."""
    data = _r.hgetall(_key(job_id))
    if not data:
        return None
    # Convert numeric fields back from strings
    for field in ("width", "height", "seed"):
        if field in data:
            try:
                data[field] = int(data[field])
            except (ValueError, TypeError):
                data[field] = None
    return data
