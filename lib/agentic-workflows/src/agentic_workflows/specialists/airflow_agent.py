"""Apache Airflow specialist agent for workflow orchestration.

Handles DAG management, task scheduling, and workflow execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import SpecialistAgent, SpecialistConfig, SpecialistCapability


@dataclass
class AirflowConfig(SpecialistConfig):
    """Airflow-specific configuration."""

    api_endpoint: str = "http://localhost:8080/api/v1"
    username: str = "airflow"
    password: str = "airflow"


class AirflowAgent(SpecialistAgent):
    """Specialist agent for Apache Airflow.

    Capabilities:
    - DAG management
    - Task scheduling
    - Workflow triggering
    - Run monitoring
    """

    def __init__(self, config: AirflowConfig | None = None, **kwargs):
        self.airflow_config = config or AirflowConfig()
        super().__init__(config=self.airflow_config, **kwargs)

        self._session = None

        self.register_handler("list_dags", self._list_dags)
        self.register_handler("get_dag", self._get_dag)
        self.register_handler("trigger_dag", self._trigger_dag)
        self.register_handler("get_dag_runs", self._get_dag_runs)
        self.register_handler("get_task_instances", self._get_task_instances)
        self.register_handler("pause_dag", self._pause_dag)
        self.register_handler("unpause_dag", self._unpause_dag)

    @property
    def capabilities(self) -> list[SpecialistCapability]:
        return [
            SpecialistCapability.WORKFLOW_ORCHESTRATION,
            SpecialistCapability.DAG_MANAGEMENT,
            SpecialistCapability.TASK_SCHEDULING,
        ]

    @property
    def service_name(self) -> str:
        return "Apache Airflow"

    async def _connect(self) -> None:
        """Connect to Airflow API."""
        import aiohttp

        auth = aiohttp.BasicAuth(self.airflow_config.username, self.airflow_config.password)
        self._session = aiohttp.ClientSession(auth=auth)

    async def _disconnect(self) -> None:
        """Disconnect from Airflow."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _health_check(self) -> bool:
        """Check Airflow health."""
        if self._session is None:
            return False
        try:
            url = f"{self.airflow_config.api_endpoint}/health"
            async with self._session.get(url) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _list_dags(
        self,
        limit: int = 100,
        offset: int = 0,
        only_active: bool = True,
    ) -> dict[str, Any]:
        """List all DAGs.

        Args:
            limit: Maximum results.
            offset: Pagination offset.
            only_active: Only show active DAGs.

        Returns:
            List of DAGs.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.airflow_config.api_endpoint}/dags"
        params = {"limit": limit, "offset": offset, "only_active": only_active}

        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to list DAGs: {resp.status}"}

    async def _get_dag(self, dag_id: str) -> dict[str, Any]:
        """Get DAG details.

        Args:
            dag_id: DAG identifier.

        Returns:
            DAG details.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.airflow_config.api_endpoint}/dags/{dag_id}"

        async with self._session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to get DAG: {resp.status}"}

    async def _trigger_dag(
        self,
        dag_id: str,
        conf: dict[str, Any] | None = None,
        execution_date: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Trigger a DAG run.

        Args:
            dag_id: DAG identifier.
            conf: Configuration for the run.
            execution_date: Execution date.
            run_id: Custom run ID.

        Returns:
            Triggered DAG run info.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.airflow_config.api_endpoint}/dags/{dag_id}/dagRuns"
        payload = {}

        if conf:
            payload["conf"] = conf
        if execution_date:
            payload["execution_date"] = execution_date
        if run_id:
            payload["dag_run_id"] = run_id

        async with self._session.post(url, json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return {"error": f"Failed to trigger DAG: {resp.status}"}

    async def _get_dag_runs(
        self,
        dag_id: str,
        limit: int = 25,
        offset: int = 0,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get DAG runs.

        Args:
            dag_id: DAG identifier.
            limit: Maximum results.
            offset: Pagination offset.
            state: Filter by state.

        Returns:
            List of DAG runs.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.airflow_config.api_endpoint}/dags/{dag_id}/dagRuns"
        params = {"limit": limit, "offset": offset}
        if state:
            params["state"] = state

        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to get DAG runs: {resp.status}"}

    async def _get_task_instances(
        self,
        dag_id: str,
        dag_run_id: str,
    ) -> dict[str, Any]:
        """Get task instances for a DAG run.

        Args:
            dag_id: DAG identifier.
            dag_run_id: DAG run identifier.

        Returns:
            List of task instances.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.airflow_config.api_endpoint}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"

        async with self._session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to get tasks: {resp.status}"}

    async def _pause_dag(self, dag_id: str) -> dict[str, Any]:
        """Pause a DAG.

        Args:
            dag_id: DAG identifier.

        Returns:
            Updated DAG info.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.airflow_config.api_endpoint}/dags/{dag_id}"
        payload = {"is_paused": True}

        async with self._session.patch(url, json=payload) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to pause DAG: {resp.status}"}

    async def _unpause_dag(self, dag_id: str) -> dict[str, Any]:
        """Unpause a DAG.

        Args:
            dag_id: DAG identifier.

        Returns:
            Updated DAG info.
        """
        if self._session is None:
            return {"error": "Not connected"}

        url = f"{self.airflow_config.api_endpoint}/dags/{dag_id}"
        payload = {"is_paused": False}

        async with self._session.patch(url, json=payload) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"error": f"Failed to unpause DAG: {resp.status}"}
