"""MinIO specialist agent for object storage operations.

Handles S3-compatible object storage for documents, media, and data assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, BinaryIO

from .base import SpecialistAgent, SpecialistCapability, SpecialistConfig


@dataclass
class MinIOConfig(SpecialistConfig):
    """MinIO-specific configuration."""

    endpoint: str = "localhost:9000"
    access_key: str = ""
    secret_key: str = ""
    secure: bool = True
    region: str = "us-east-1"
    default_bucket: str = "agentic-workflows"


class MinIOAgent(SpecialistAgent):
    """Specialist agent for MinIO object storage.

    Capabilities:
    - Object upload/download
    - Bucket management
    - Pre-signed URL generation
    - Object metadata management
    - Versioning support
    """

    def __init__(self, config: MinIOConfig | None = None, **kwargs):
        """Initialize MinIO agent.

        Args:
            config: MinIO configuration.
            **kwargs: Additional agent arguments.
        """
        self.minio_config = config or MinIOConfig()
        super().__init__(config=self.minio_config, **kwargs)

        self._client = None

        # Register handlers
        self.register_handler("upload_object", self._upload_object)
        self.register_handler("download_object", self._download_object)
        self.register_handler("delete_object", self._delete_object)
        self.register_handler("list_objects", self._list_objects)
        self.register_handler("get_presigned_url", self._get_presigned_url)
        self.register_handler("create_bucket", self._create_bucket)
        self.register_handler("list_buckets", self._list_buckets)
        self.register_handler("get_object_info", self._get_object_info)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.OBJECT_STORAGE,
            SpecialistCapability.BLOB_MANAGEMENT,
        ]

    @property
    def service_name(self) -> str:
        return "MinIO"

    async def _connect(self) -> None:
        """Connect to MinIO server."""
        try:
            from minio import Minio

            self._client = Minio(
                self.minio_config.endpoint,
                access_key=self.minio_config.access_key,
                secret_key=self.minio_config.secret_key,
                secure=self.minio_config.secure,
                region=self.minio_config.region,
            )

            # Ensure default bucket exists
            if not self._client.bucket_exists(self.minio_config.default_bucket):
                self._client.make_bucket(self.minio_config.default_bucket)

        except ImportError:
            self.logger.warning("minio package not installed, using mock client")
            self._client = None

    async def _disconnect(self) -> None:
        """Disconnect from MinIO."""
        self._client = None

    async def _health_check(self) -> bool:
        """Check MinIO health."""
        if self._client is None:
            return False

        try:
            self._client.list_buckets()
            return True
        except Exception:
            return False

    async def _upload_object(
        self,
        bucket: str | None = None,
        object_name: str = "",
        data: bytes | BinaryIO = b"",
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Upload an object to MinIO.

        Args:
            bucket: Bucket name (uses default if not specified).
            object_name: Object key/name.
            data: Object data.
            content_type: MIME type.
            metadata: Optional metadata.

        Returns:
            Upload result with etag and version.
        """
        bucket = bucket or self.minio_config.default_bucket

        if self._client is None:
            return {"error": "Client not connected"}

        import io

        if isinstance(data, bytes):
            data_stream = io.BytesIO(data)
            length = len(data)
        else:
            data_stream = data
            data_stream.seek(0, 2)
            length = data_stream.tell()
            data_stream.seek(0)

        result = self._client.put_object(
            bucket,
            object_name,
            data_stream,
            length,
            content_type=content_type,
            metadata=metadata,
        )

        return {
            "bucket": bucket,
            "object_name": object_name,
            "etag": result.etag,
            "version_id": result.version_id,
        }

    async def _download_object(
        self,
        bucket: str | None = None,
        object_name: str = "",
        version_id: str | None = None,
    ) -> dict[str, Any]:
        """Download an object from MinIO.

        Args:
            bucket: Bucket name.
            object_name: Object key/name.
            version_id: Optional version ID.

        Returns:
            Object data and metadata.
        """
        bucket = bucket or self.minio_config.default_bucket

        if self._client is None:
            return {"error": "Client not connected"}

        response = self._client.get_object(bucket, object_name, version_id=version_id)

        try:
            data = response.read()
            return {
                "bucket": bucket,
                "object_name": object_name,
                "data": data,
                "content_type": response.headers.get("Content-Type"),
                "size": len(data),
            }
        finally:
            response.close()
            response.release_conn()

    async def _delete_object(
        self,
        bucket: str | None = None,
        object_name: str = "",
        version_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete an object from MinIO.

        Args:
            bucket: Bucket name.
            object_name: Object key/name.
            version_id: Optional version ID.

        Returns:
            Deletion confirmation.
        """
        bucket = bucket or self.minio_config.default_bucket

        if self._client is None:
            return {"error": "Client not connected"}

        self._client.remove_object(bucket, object_name, version_id=version_id)

        return {
            "bucket": bucket,
            "object_name": object_name,
            "deleted": True,
        }

    async def _list_objects(
        self,
        bucket: str | None = None,
        prefix: str = "",
        recursive: bool = True,
        max_keys: int = 1000,
    ) -> list[dict[str, Any]]:
        """List objects in a bucket.

        Args:
            bucket: Bucket name.
            prefix: Object prefix filter.
            recursive: Include subdirectories.
            max_keys: Maximum objects to return.

        Returns:
            List of object info.
        """
        bucket = bucket or self.minio_config.default_bucket

        if self._client is None:
            return []

        objects = self._client.list_objects(
            bucket,
            prefix=prefix,
            recursive=recursive,
        )

        result = []
        for obj in objects:
            if len(result) >= max_keys:
                break
            result.append(
                {
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "etag": obj.etag,
                    "is_dir": obj.is_dir,
                }
            )

        return result

    async def _get_presigned_url(
        self,
        bucket: str | None = None,
        object_name: str = "",
        expires: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate a pre-signed URL for an object.

        Args:
            bucket: Bucket name.
            object_name: Object key/name.
            expires: URL expiration in seconds.
            method: HTTP method (GET or PUT).

        Returns:
            Pre-signed URL.
        """
        from datetime import timedelta

        bucket = bucket or self.minio_config.default_bucket

        if self._client is None:
            return ""

        if method.upper() == "PUT":
            return self._client.presigned_put_object(
                bucket,
                object_name,
                expires=timedelta(seconds=expires),
            )
        else:
            return self._client.presigned_get_object(
                bucket,
                object_name,
                expires=timedelta(seconds=expires),
            )

    async def _create_bucket(
        self,
        bucket: str,
        region: str | None = None,
    ) -> dict[str, Any]:
        """Create a new bucket.

        Args:
            bucket: Bucket name.
            region: Bucket region.

        Returns:
            Creation result.
        """
        if self._client is None:
            return {"error": "Client not connected"}

        if self._client.bucket_exists(bucket):
            return {"bucket": bucket, "created": False, "exists": True}

        self._client.make_bucket(bucket, location=region or self.minio_config.region)

        return {"bucket": bucket, "created": True}

    async def _list_buckets(self) -> list[dict[str, Any]]:
        """List all buckets.

        Returns:
            List of bucket info.
        """
        if self._client is None:
            return []

        buckets = self._client.list_buckets()

        return [
            {
                "name": b.name,
                "creation_date": b.creation_date.isoformat() if b.creation_date else None,
            }
            for b in buckets
        ]

    async def _get_object_info(
        self,
        bucket: str | None = None,
        object_name: str = "",
    ) -> dict[str, Any]:
        """Get object metadata.

        Args:
            bucket: Bucket name.
            object_name: Object key/name.

        Returns:
            Object metadata.
        """
        bucket = bucket or self.minio_config.default_bucket

        if self._client is None:
            return {"error": "Client not connected"}

        stat = self._client.stat_object(bucket, object_name)

        return {
            "bucket": bucket,
            "object_name": object_name,
            "size": stat.size,
            "etag": stat.etag,
            "content_type": stat.content_type,
            "last_modified": stat.last_modified.isoformat() if stat.last_modified else None,
            "version_id": stat.version_id,
            "metadata": dict(stat.metadata) if stat.metadata else {},
        }
