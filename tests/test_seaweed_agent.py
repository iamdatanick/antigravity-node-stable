import asyncio
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from agentic_workflows.specialists.seaweed_agent import SeaweedAgent


def _not_found_error():
    return ClientError(
        error_response={"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
        operation_name="HeadBucket",
    )


@pytest.mark.asyncio
async def test_initialize_creates_bucket_when_missing():
    mock_client = MagicMock()
    mock_client.head_bucket.side_effect = _not_found_error()

    with patch("agentic_workflows.specialists.seaweed_agent.boto3.client", return_value=mock_client):
        agent = SeaweedAgent()
        await agent.initialize()

        mock_client.create_bucket.assert_called_once_with(Bucket=agent.seaweed_config.default_bucket)


@pytest.mark.asyncio
async def test_upload_puts_object():
    mock_client = MagicMock()
    mock_client.head_bucket.return_value = {}

    with patch("agentic_workflows.specialists.seaweed_agent.boto3.client", return_value=mock_client):
        agent = SeaweedAgent()
        await agent.initialize()
        mock_client.create_bucket.reset_mock()

        result = await agent._upload_object(object_name="file.txt", data=b"data")

        mock_client.put_object.assert_called_once()
        assert result["object_name"] == "file.txt"
