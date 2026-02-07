"""SeaweedFS S3 client wrapper using boto3."""

import os
import logging
import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger("antigravity.s3")

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "admin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "admin")
S3_BUCKET = os.environ.get("S3_BUCKET", "antigravity")


def get_client():
    """Create a boto3 S3 client pointing to SeaweedFS."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(bucket: str = S3_BUCKET):
    """Create a bucket if it does not exist."""
    client = get_client()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)
        logger.info(f"Created S3 bucket: {bucket}")


def upload(key: str, data: bytes, bucket: str = S3_BUCKET):
    """Upload bytes to SeaweedFS S3."""
    ensure_bucket(bucket)
    client = get_client()
    client.put_object(Bucket=bucket, Key=key, Body=data)
    logger.info(f"Uploaded s3://{bucket}/{key} ({len(data)} bytes)")


def download(key: str, bucket: str = S3_BUCKET) -> bytes:
    """Download bytes from SeaweedFS S3."""
    client = get_client()
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def list_objects(prefix: str = "", bucket: str = S3_BUCKET) -> list:
    """List object keys with optional prefix filter."""
    client = get_client()
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]
