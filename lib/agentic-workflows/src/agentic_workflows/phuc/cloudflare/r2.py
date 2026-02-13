"""R2 object storage for PHUC documents."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Optional, Union
import httpx
import os
import hashlib
from datetime import datetime


@dataclass
class R2Config:
    """R2 bucket configuration."""
    account_id: str = field(default_factory=lambda: os.getenv("CF_ACCOUNT_ID", ""))
    api_token: str = field(default_factory=lambda: os.getenv("CF_API_TOKEN", ""))
    bucket_name: str = "phucai"
    
    # S3-compatible access
    access_key_id: str = field(default_factory=lambda: os.getenv("R2_ACCESS_KEY_ID", ""))
    secret_access_key: str = field(default_factory=lambda: os.getenv("R2_SECRET_ACCESS_KEY", ""))


@dataclass
class R2Object:
    """R2 object metadata."""
    key: str
    size: int
    etag: str
    last_modified: datetime
    content_type: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class R2Storage:
    """R2 object storage client."""
    
    BASE_URL = "https://api.cloudflare.com/client/v4"
    SUPPORTED_FORMATS = ["txt", "pdf", "csv", "json", "png", "jpg", "jpeg", "xlsx", "docx"]
    MAX_FILE_SIZE = 104857600  # 100MB
    
    def __init__(self, config: R2Config = None):
        self.config = config or R2Config()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_token}",
        }
    
    @property
    def base_url(self) -> str:
        return f"{self.BASE_URL}/accounts/{self.config.account_id}/r2/buckets/{self.config.bucket_name}"
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers, timeout=60.0)
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _validate_file(self, filename: str, content: bytes) -> tuple[bool, str]:
        """Validate file before upload."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        
        if ext not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported format: {ext}"
        
        if len(content) > self.MAX_FILE_SIZE:
            return False, f"File too large: {len(content)} > {self.MAX_FILE_SIZE}"
        
        return True, ""
    
    def _generate_key(self, user_id: str, filename: str) -> str:
        """Generate unique R2 key for file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"users/{user_id}/documents/{timestamp}_{file_hash}_{filename}"
    
    async def upload(
        self,
        key: str,
        content: Union[bytes, BinaryIO],
        content_type: str = "application/octet-stream",
        metadata: dict = None
    ) -> R2Object:
        """Upload object to R2."""
        client = await self._get_client()
        
        if hasattr(content, "read"):
            content = content.read()
        
        headers = {
            **self.headers,
            "Content-Type": content_type
        }
        
        if metadata:
            for k, v in metadata.items():
                headers[f"x-amz-meta-{k}"] = str(v)
        
        response = await client.put(
            f"{self.base_url}/objects/{key}",
            content=content,
            headers=headers
        )
        response.raise_for_status()
        
        return R2Object(
            key=key,
            size=len(content),
            etag=response.headers.get("etag", ""),
            last_modified=datetime.utcnow(),
            content_type=content_type,
            metadata=metadata or {}
        )
    
    async def upload_document(
        self,
        user_id: str,
        filename: str,
        content: bytes,
        content_type: str = None
    ) -> R2Object:
        """Upload document with validation."""
        valid, error = self._validate_file(filename, content)
        if not valid:
            raise ValueError(error)
        
        key = self._generate_key(user_id, filename)
        
        if not content_type:
            ext = filename.rsplit(".", 1)[-1].lower()
            content_type = self._get_mime_type(ext)
        
        return await self.upload(
            key=key,
            content=content,
            content_type=content_type,
            metadata={
                "user_id": user_id,
                "original_filename": filename
            }
        )
    
    def _get_mime_type(self, ext: str) -> str:
        """Get MIME type for extension."""
        types = {
            "txt": "text/plain",
            "pdf": "application/pdf",
            "csv": "text/csv",
            "json": "application/json",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }
        return types.get(ext, "application/octet-stream")
    
    async def download(self, key: str) -> bytes:
        """Download object from R2."""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/objects/{key}")
        response.raise_for_status()
        return response.content
    
    async def get_metadata(self, key: str) -> Optional[R2Object]:
        """Get object metadata."""
        client = await self._get_client()
        response = await client.head(f"{self.base_url}/objects/{key}")
        
        if response.status_code != 200:
            return None
        
        return R2Object(
            key=key,
            size=int(response.headers.get("content-length", 0)),
            etag=response.headers.get("etag", ""),
            last_modified=datetime.utcnow(),  # Parse from header if available
            content_type=response.headers.get("content-type")
        )
    
    async def list_objects(
        self,
        prefix: str = "",
        limit: int = 1000,
        cursor: str = None
    ) -> tuple[list[R2Object], Optional[str]]:
        """List objects in bucket."""
        client = await self._get_client()
        
        params = {"limit": limit}
        if prefix:
            params["prefix"] = prefix
        if cursor:
            params["cursor"] = cursor
        
        response = await client.get(f"{self.base_url}/objects", params=params)
        response.raise_for_status()
        
        data = response.json()
        objects = []
        
        for obj in data.get("result", {}).get("objects", []):
            objects.append(R2Object(
                key=obj["key"],
                size=obj["size"],
                etag=obj.get("etag", ""),
                last_modified=datetime.fromisoformat(obj["uploaded"].replace("Z", "+00:00"))
            ))
        
        next_cursor = data.get("result", {}).get("truncated") and data.get("result", {}).get("cursor")
        return objects, next_cursor
    
    async def list_user_documents(self, user_id: str) -> list[R2Object]:
        """List all documents for a user."""
        objects, _ = await self.list_objects(prefix=f"users/{user_id}/documents/")
        return objects
    
    async def delete(self, key: str) -> bool:
        """Delete object from R2."""
        client = await self._get_client()
        response = await client.delete(f"{self.base_url}/objects/{key}")
        return response.status_code in (200, 204)
    
    async def delete_user_documents(self, user_id: str) -> int:
        """Delete all documents for a user."""
        objects = await self.list_user_documents(user_id)
        deleted = 0
        for obj in objects:
            if await self.delete(obj.key):
                deleted += 1
        return deleted
