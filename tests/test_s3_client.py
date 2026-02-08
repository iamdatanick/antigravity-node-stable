"""Tests for s3_client.py S3 operations."""

from unittest.mock import MagicMock, patch


class TestS3Client:
    """Test S3 client functions."""

    @patch("workflows.s3_client.boto3.client")
    def test_singleton_client(self, mock_boto3_client):
        """Test that get_client returns a boto3 S3 client."""
        from workflows.s3_client import get_client

        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        get_client()

        # Verify boto3.client was called with correct parameters
        assert mock_boto3_client.called
        call_kwargs = mock_boto3_client.call_args[1]
        assert call_kwargs["endpoint_url"] is not None
        assert "aws_access_key_id" in call_kwargs
        assert "aws_secret_access_key" in call_kwargs


class TestUpload:
    """Test S3 upload function."""

    @patch("workflows.s3_client.ensure_bucket")
    @patch("workflows.s3_client.get_client")
    def test_upload(self, mock_get_client, mock_ensure_bucket):
        """Test upload calls put_object with correct parameters."""
        from workflows.s3_client import upload

        mock_s3 = MagicMock()
        mock_get_client.return_value = mock_s3
        mock_ensure_bucket.return_value = None

        # Call upload
        upload("test-key", b"test data", bucket="test-bucket")

        # Verify put_object was called
        assert mock_s3.put_object.called
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "test-key"
        assert call_kwargs["Body"] == b"test data"


class TestDownload:
    """Test S3 download function."""

    @patch("workflows.s3_client.get_client")
    def test_download(self, mock_get_client):
        """Test download calls get_object and reads body."""
        from workflows.s3_client import download

        mock_s3 = MagicMock()
        mock_body = MagicMock()
        mock_body.read.return_value = b"downloaded data"
        mock_s3.get_object.return_value = {"Body": mock_body}
        mock_get_client.return_value = mock_s3

        # Call download
        result = download("test-key", bucket="test-bucket")

        # Verify get_object was called
        assert mock_s3.get_object.called
        call_kwargs = mock_s3.get_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "test-key"

        # Verify result
        assert result == b"downloaded data"


class TestEnsureBucket:
    """Test ensure_bucket function."""

    @patch("workflows.s3_client.get_client")
    def test_ensure_bucket_creates_if_missing(self, mock_get_client):
        """Test ensure_bucket creates bucket if head_bucket raises exception."""
        from workflows.s3_client import _bucket_ensured, ensure_bucket

        _bucket_ensured.discard("test-bucket")

        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        # Simulate bucket doesn't exist — must raise ClientError (not plain Exception)
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
        mock_s3.head_bucket.side_effect = ClientError(error_response, "HeadBucket")
        mock_get_client.return_value = mock_s3

        # Call ensure_bucket
        ensure_bucket("test-bucket")

        # Verify create_bucket was called
        assert mock_s3.create_bucket.called
        call_kwargs = mock_s3.create_bucket.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"

    @patch("workflows.s3_client.get_client")
    def test_ensure_bucket_caches(self, mock_get_client):
        """Test ensure_bucket doesn't recreate existing bucket."""
        from workflows.s3_client import _bucket_ensured, ensure_bucket

        _bucket_ensured.discard("test-bucket")

        mock_s3 = MagicMock()
        # Simulate bucket exists
        mock_s3.head_bucket.return_value = {}
        mock_get_client.return_value = mock_s3

        # Call ensure_bucket
        ensure_bucket("test-bucket")

        # Verify head_bucket was called but create_bucket was not
        assert mock_s3.head_bucket.called
        assert not mock_s3.create_bucket.called
