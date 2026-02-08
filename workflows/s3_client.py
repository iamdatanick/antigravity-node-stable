"""SeaweedFS S3 client wrapper using boto3."""

import logging
import os
import threading

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from workflows.telemetry import get_tracer

logger = logging.getLogger("antigravity.s3")
tracer = get_tracer("antigravity.s3")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "admin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "admin")
S3_BUCKET = os.environ.get("S3_BUCKET", "antigravity")

_s3_client = None
_s3_client_lock = threading.Lock()
_bucket_ensured = set()
_bucket_lock = threading.Lock()


def get_client():
    """Get or create a singleton boto3 S3 client pointing to SeaweedFS."""
    global _s3_client
    if _s3_client is None:
        with _s3_client_lock:
            # Double-check locking pattern
            if _s3_client is None:
                _s3_client = boto3.client(
                    "s3",
                    endpoint_url=S3_ENDPOINT,
                    aws_access_key_id=S3_ACCESS_KEY,
                    aws_secret_access_key=S3_SECRET_KEY,
                    config=BotoConfig(signature_version="s3v4"),
                    region_name="us-east-1",
                )
    return _s3_client


def ensure_bucket(bucket: str = S3_BUCKET):
    """Create a bucket if it does not exist (cached after first check)."""
    with _bucket_lock:
        if bucket in _bucket_ensured:
            return
        client = get_client()
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError as e:
            # Check the error code - it can be a string like '404' or 'NoSuchBucket'
            error_code = e.response.get("Error", {}).get("Code", "")
            http_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if error_code in ("404", "NoSuchBucket") or http_status == 404:
                client.create_bucket(Bucket=bucket)
                logger.info(f"Created S3 bucket: {bucket}")
            else:
                raise
        _bucket_ensured.add(bucket)


def upload(key: str, data: bytes, bucket: str = S3_BUCKET):
    """Upload bytes to SeaweedFS S3."""
    with tracer.start_as_current_span("s3.upload", attributes={"key": key, "bucket": bucket, "size_bytes": len(data)}):
        ensure_bucket(bucket)
        client = get_client()
        client.put_object(Bucket=bucket, Key=key, Body=data)
        logger.info(f"Uploaded s3://{bucket}/{key} ({len(data)} bytes)")


def download(key: str, bucket: str = S3_BUCKET) -> bytes:
    """Download bytes from SeaweedFS S3."""
    with tracer.start_as_current_span("s3.download", attributes={"key": key, "bucket": bucket}) as span:
        client = get_client()
        resp = client.get_object(Bucket=bucket, Key=key)
        # Protect against memory explosion (limit 100MB for direct read)
        body = resp["Body"]
        # Note: ContentLength might be missing if chunked, but usually present for S3
        content_len = resp.get("ContentLength", 0)

        # 100MB Limit
        max_size = 100 * 1024 * 1024

        if content_len > max_size:
            logger.warning(f"Large file download ({content_len} bytes) - truncating to 100MB")
            data = body.read(max_size)
        else:
            # Read all, but check size during read if possible (boto3 doesn't easily support limit on read())
            # So we rely on ContentLength check mainly.
            data = body.read()
            if len(data) > max_size:
                data = data[:max_size]
                logger.warning(f"Large file download (actual {len(data)} bytes) - truncated to 100MB")
        span.set_attribute("size_bytes", len(data))
        logger.info(f"Downloaded s3://{bucket}/{key} ({len(data)} bytes)")
        return data


def list_objects(prefix: str = "", bucket: str = S3_BUCKET) -> list:
    """List object keys with optional prefix filter."""
    client = get_client()
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]
