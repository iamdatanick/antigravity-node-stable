"""OpenVINO Model Server (OVMS) specialist agent for ML inference.

Handles model serving, inference, and model management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability


@dataclass
class OVMSConfig(SpecialistConfig):
    """OVMS-specific configuration."""

    grpc_endpoint: str = "localhost:9000"
    rest_endpoint: str = "http://localhost:8000"
    model_name: str = "default"
    model_version: int | None = None


class OVMSAgent(SpecialistAgent):
    """Specialist agent for OpenVINO Model Server.

    Capabilities:
    - Model inference
    - Model management
    - Batch predictions
    - Model metadata
    """

    def __init__(self, config: OVMSConfig | None = None, **kwargs):
        self.ovms_config = config or OVMSConfig()
        super().__init__(config=self.ovms_config, **kwargs)

        self._session = None

        self.register_handler("predict", self._predict)
        self.register_handler("batch_predict", self._batch_predict)
        self.register_handler("get_model_status", self._get_model_status)
        self.register_handler("get_model_metadata", self._get_model_metadata)
        self.register_handler("list_models", self._list_models)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.MODEL_SERVING,
            SpecialistCapability.INFERENCE,
            SpecialistCapability.MODEL_MANAGEMENT,
        ]

    @property
    def service_name(self) -> str:
        return "OpenVINO Model Server"

    async def _connect(self) -> None:
        """Connect to OVMS."""
        import aiohttp
        self._session = aiohttp.ClientSession()

    async def _disconnect(self) -> None:
        """Disconnect from OVMS."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _health_check(self) -> bool:
        """Check OVMS health."""
        if self._session is None:
            return False
        try:
            url = f"{self.ovms_config.rest_endpoint}/v1/config"
            async with self._session.get(url) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _predict(
        self,
        model_name: str | None = None,
        inputs: dict[str, Any] = None,
        model_version: int | None = None,
    ) -> dict[str, Any]:
        """Run inference on a model.

        Args:
            model_name: Model name.
            inputs: Model inputs.
            model_version: Specific model version.

        Returns:
            Prediction results.
        """
        if self._session is None:
            return {"error": "Not connected"}

        model = model_name or self.ovms_config.model_name
        version = model_version or self.ovms_config.model_version

        if version:
            url = f"{self.ovms_config.rest_endpoint}/v1/models/{model}/versions/{version}:predict"
        else:
            url = f"{self.ovms_config.rest_endpoint}/v1/models/{model}:predict"

        payload = {"inputs": inputs}

        async with self._session.post(url, json=payload) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Prediction failed: {resp.status}"}

    async def _batch_predict(
        self,
        model_name: str | None = None,
        batch_inputs: list[dict[str, Any]] = None,
        model_version: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run batch inference.

        Args:
            model_name: Model name.
            batch_inputs: List of model inputs.
            model_version: Specific model version.

        Returns:
            List of prediction results.
        """
        results = []
        for inputs in batch_inputs or []:
            result = await self._predict(model_name, inputs, model_version)
            results.append(result)
        return results

    async def _get_model_status(
        self,
        model_name: str | None = None,
        model_version: int | None = None,
    ) -> dict[str, Any]:
        """Get model status.

        Args:
            model_name: Model name.
            model_version: Specific model version.

        Returns:
            Model status.
        """
        if self._session is None:
            return {"error": "Not connected"}

        model = model_name or self.ovms_config.model_name

        if model_version:
            url = f"{self.ovms_config.rest_endpoint}/v1/models/{model}/versions/{model_version}"
        else:
            url = f"{self.ovms_config.rest_endpoint}/v1/models/{model}"

        async with self._session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to get status: {resp.status}"}

    async def _get_model_metadata(
        self,
        model_name: str | None = None,
        model_version: int | None = None,
    ) -> dict[str, Any]:
        """Get model metadata including input/output specs.

        Args:
            model_name: Model name.
            model_version: Specific model version.

        Returns:
            Model metadata.
        """
        if self._session is None:
            return {"error": "Not connected"}

        model = model_name or self.ovms_config.model_name

        if model_version:
            url = f"{self.ovms_config.rest_endpoint}/v1/models/{model}/versions/{model_version}/metadata"
        else:
            url = f"{self.ovms_config.rest_endpoint}/v1/models/{model}/metadata"

        async with self._session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to get metadata: {resp.status}"}

    async def _list_models(self) -> list[dict[str, Any]]:
        """List all available models.

        Returns:
            List of model info.
        """
        if self._session is None:
            return []

        url = f"{self.ovms_config.rest_endpoint}/v1/config"

        async with self._session.get(url) as resp:
            if resp.status == 200:
                config = await resp.json()
                models = []
                for model_config in config.get("model_config_list", []):
                    models.append({
                        "name": model_config.get("config", {}).get("name"),
                        "base_path": model_config.get("config", {}).get("base_path"),
                    })
                return models
            return []
