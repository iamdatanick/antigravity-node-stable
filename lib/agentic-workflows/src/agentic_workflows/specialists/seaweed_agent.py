"""SeaweedFS specialist agent for S3-compatible object storage."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, BinaryIO

import boto3
from botocore.exceptions import ClientError

from .base import SpecialistAgent, SpecialistCapability, SpecialistConfig

DEFAULT_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", os.environ.get("S3_ENDPOINT", "http://seaweedfs:8333"))
DEFAULT_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", os.environ.get("S3_ACCESS_KEY", "antigravity"))
DEFAULT_SECRET_KEY = os.environ.get(
    "AWS_SECRET_ACCESS_KEY", os.environ.get("S3_SECRET_KEY", "antigravity_secret")
)
DEFAULT_BUCKET = os.environ.get("S3_BUCKET", "antigravity")


@dataclass
class SeaweedConfig(SpecialistConfig):
    """SeaweedFS configuration."""

    name: str = "seaweedfs"
    endpoint: str = DEFAULT_ENDPOINT
    access_key: str = DEFAULT_ACCESS_KEY
    secret_key: str = DEFAULT_SECRET_KEY
    region: str = "us-east-1"
    default_bucket: str = DEFAULT_BUCKET


class SeaweedAgent(SpecialistAgent):
    """Specialist agent for SeaweedFS object storage."""

    def __init__(self, config: SeaweedConfig | None = None, **kwargs):
        self.seaweed_config = config or SeaweedConfig()
        super().__init__(config=self.seaweed_config, **kwargs)

        self._client = None

        self.register_handler("upload_object", self._upload_object)
        self.register_handler("download_object", self._download_object)
        self.register_handler("delete_object", self._delete_object)
        self.register_handler("list_objects", self._list_objects)
        self.register_handler("get_presigned_url", self._get_presigned_url)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.OBJECT_STORAGE,
            SpecialistCapability.BLOB_MANAGEMENT,
        ]

    @property
    def service_name(self) -> str:
        return "SeaweedFS"

    async def _connect(self) -> None:
        """Connect to SeaweedFS S3 gateway and ensure default bucket exists."""
        self._client = boto3.client(
            "s3",
            endpoint_url=self.seaweed_config.endpoint,
            aws_access_key_id=self.seaweed_config.access_key,
            aws_secret_access_key=self.seaweed_config.secret_key,
            region_name=self.seaweed_config.region,
        )

        try:
            self._client.head_bucket(Bucket=self.seaweed_config.default_bucket)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            http_status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if error_code in ("404", "NoSuchBucket") or http_status == 404:
                self._client.create_bucket(Bucket=self.seaweed_config.default_bucket)
            else:
                raise

    async def _disconnect(self) -> None:
        """Disconnect from SeaweedFS."""
        self._client = None

    async def _health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.list_buckets()
            return True
        except Exception:
            return False

    def _ensure_client(self):
        if self._client is None:
            raise RuntimeError("SeaweedFS client not initialized. Call initialize() first.")
        return self._client

    async def _upload_object(
        self,
        bucket: str | None = None,
        object_name: str = "",
        data: bytes | BinaryIO = b"",
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        bucket = bucket or self.seaweed_config.default_bucket
        client = self._ensure_client()

        if isinstance(data, bytes):
            stream = io.BytesIO(data)
        else:
            stream = data
            stream.seek(0)

        client.put_object(
            Bucket=bucket,
            Key=object_name,
            Body=stream,
            ContentType=content_type,
            Metadata=metadata,
        )

        return {"bucket": bucket, "object_name": object_name}

    async def _download_object(
        self,
        bucket: str | None = None,
        object_name: str = "",
    ) -> dict[str, Any]:
        bucket = bucket or self.seaweed_config.default_bucket
        client = self._ensure_client()

        resp = client.get_object(Bucket=bucket, Key=object_name)
        data = resp["Body"].read()
        return {
            "bucket": bucket,
            "object_name": object_name,
            "data": data,
            "content_type": resp.get("ContentType"),
            "size": len(data),
        }

    async def _delete_object(
        self,
        bucket: str | None = None,
        object_name: str = "",
    ) -> dict[str, Any]:
        bucket = bucket or self.seaweed_config.default_bucket
        client = self._ensure_client()

        client.delete_object(Bucket=bucket, Key=object_name)
        return {"bucket": bucket, "object_name": object_name, "deleted": True}

    async def _list_objects(
        self,
        bucket: str | None = None,
        prefix: str = "",
    ) -> dict[str, Any]:
        bucket = bucket or self.seaweed_config.default_bucket
        client = self._ensure_client()

        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return {
            "bucket": bucket,
            "objects": [obj["Key"] for obj in resp.get("Contents", [])],
        }

    async def _get_presigned_url(
        self,
        bucket: str | None = None,
        object_name: str = "",
        expires_in: int = 3600,
    ) -> dict[str, Any]:
        bucket = bucket or self.seaweed_config.default_bucket
        client = self._ensure_client()

        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": object_name},
            ExpiresIn=expires_in,
        )
        return {"bucket": bucket, "object_name": object_name, "url": url}

    async def _execute(self, input_data: Any) -> Any:
        """Default execute simply returns available operations."""
        return {
            "service": self.service_name,
            "bucket": self.seaweed_config.default_bucket,
            "operations": list(self._handlers.keys()),
            "input": input_data,
        }
