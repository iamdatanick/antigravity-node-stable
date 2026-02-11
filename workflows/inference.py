"""OVMS inference pipeline for Antigravity Node v14.1.

Connects to OpenVINO Model Server via gRPC (primary) and REST (fallback).
Provides run_inference() with graceful degradation when no model is loaded,
and OpenTelemetry span wrapping for all inference calls.

Environment variables:
    OVMS_GRPC      -- gRPC target (default: ovms:9000)
    OVMS_REST_URL  -- REST base URL (default: http://ovms:9001)
    OVMS_REST      -- Legacy REST env var (fallback)
"""

import logging
import os
import time
from typing import Any

import grpc
import grpc.aio
import httpx
from opentelemetry import trace

logger = logging.getLogger("antigravity.inference")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OVMS_GRPC_TARGET = os.environ.get("OVMS_GRPC", "ovms:9000")
OVMS_REST_BASE = os.environ.get("OVMS_REST_URL", os.environ.get("OVMS_REST", "http://ovms:9001"))
GRPC_TIMEOUT_S = float(os.environ.get("OVMS_GRPC_TIMEOUT", "10"))
REST_TIMEOUT_S = float(os.environ.get("OVMS_REST_TIMEOUT", "10"))

# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------
tracer = trace.get_tracer("antigravity.inference")


# ---------------------------------------------------------------------------
# gRPC helpers (lazy-loaded to avoid import failure at startup)
# ---------------------------------------------------------------------------
_grpc_channel: grpc.aio.Channel | None = None


def _get_predict_service_stub():
    """Lazy-import tensorflow_serving predict service stub.

    The grpcio-generated stubs for TFS/OVMS predict API are produced by
    ovmsclient or tensorflow-serving-api.  We try ovmsclient first (lighter),
    then tensorflow-serving-api, then fall back to REST.
    """
    # Try ovmsclient (pip install ovmsclient)
    try:
        from ovmsclient.tfs_compat.grpc.tensors import (  # type: ignore[import-untyped]
            make_tensor_proto,
        )
        from tensorflow_serving.apis import (  # type: ignore[import-untyped]
            predict_pb2,
            prediction_service_pb2_grpc,
        )

        return prediction_service_pb2_grpc, predict_pb2, make_tensor_proto
    except ImportError:
        pass

    # Try tensorflow-serving-api standalone
    try:
        from tensorflow_serving.apis import (  # type: ignore[import-untyped]
            predict_pb2,
            prediction_service_pb2_grpc,
        )

        return prediction_service_pb2_grpc, predict_pb2, None
    except ImportError:
        pass

    return None, None, None


async def _ensure_grpc_channel() -> grpc.aio.Channel | None:
    """Create or reuse an async gRPC channel to OVMS."""
    global _grpc_channel
    if _grpc_channel is None:
        try:
            _grpc_channel = grpc.aio.insecure_channel(
                OVMS_GRPC_TARGET,
                options=[
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 30000),
                ],
            )
        except Exception as exc:
            logger.warning("Failed to create gRPC channel to OVMS: %s", exc)
            _grpc_channel = None
    return _grpc_channel


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
async def ovms_health_check() -> dict[str, Any]:
    """Check whether OVMS is reachable and serving.

    Uses the REST /v1/config endpoint (same as docker-compose healthcheck).
    Returns dict compatible with health.py check format.
    """
    with tracer.start_as_current_span("ovms.health_check") as span:
        span.set_attribute("ovms.rest_base", OVMS_REST_BASE)
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{OVMS_REST_BASE}/v1/config")
                config = resp.json() if resp.status_code == 200 else {}
                model_count = len(config.get("model_config_list", []))
                span.set_attribute("ovms.model_count", model_count)
                return {
                    "name": "ovms",
                    "healthy": True,
                    "error": None,
                    "model_count": model_count,
                }
        except Exception as exc:
            span.set_attribute("ovms.error", str(exc))
            return {
                "name": "ovms",
                "healthy": False,
                "error": str(exc),
                "model_count": 0,
            }


async def list_models() -> list[str]:
    """Return list of model names currently loaded in OVMS."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OVMS_REST_BASE}/v1/config")
            if resp.status_code == 200:
                config = resp.json()
                models = []
                for entry in config.get("model_config_list", []):
                    cfg = entry.get("config", {})
                    name = cfg.get("name", "")
                    if name:
                        models.append(name)
                return models
    except Exception as exc:
        logger.warning("Failed to list OVMS models: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------
async def run_inference(model_name: str, input_data: dict) -> dict:
    """Run inference against an OVMS model.

    Parameters
    ----------
    model_name : str
        Name of the model deployed in OVMS (as in its config).
    input_data : dict
        Mapping of input tensor names to values.
        For REST, values should be JSON-serialisable (lists/nested lists of numbers).
        Example: {"input": [[1.0, 2.0, 3.0]]}

    Returns
    -------
    dict
        On success:   {"status": "ok", "model": ..., "outputs": {...}, "latency_ms": float}
        On no model:  {"status": "no_model_loaded", "message": "..."}
        On not found: {"status": "model_not_found", "message": "..."}
        On error:     {"status": "error", "message": "..."}
    """
    with tracer.start_as_current_span("ovms.run_inference") as span:
        span.set_attribute("ovms.model_name", model_name)
        t0 = time.monotonic()

        # 1. Quick model existence check (via REST config)
        available = await list_models()
        span.set_attribute("ovms.available_models", str(available))

        if not available:
            msg = (
                "OVMS has no models loaded. Deploy a model to "
                "the ./models/ volume and OVMS will auto-detect it "
                "(file_system_poll_wait_seconds=5)."
            )
            logger.info("run_inference(%s): no models loaded", model_name)
            span.set_attribute("ovms.result", "no_model_loaded")
            return {"status": "no_model_loaded", "message": msg}

        if model_name not in available:
            msg = f"Model '{model_name}' not found. Available models: {available}"
            logger.info("run_inference(%s): model not found", model_name)
            span.set_attribute("ovms.result", "model_not_found")
            return {"status": "model_not_found", "message": msg}

        # 2. Attempt gRPC inference (primary path)
        grpc_result = await _try_grpc_inference(model_name, input_data, span)
        if grpc_result is not None:
            latency = (time.monotonic() - t0) * 1000
            grpc_result["latency_ms"] = round(latency, 2)
            span.set_attribute("ovms.latency_ms", latency)
            span.set_attribute("ovms.transport", "grpc")
            return grpc_result

        # 3. Fallback to REST inference
        rest_result = await _try_rest_inference(model_name, input_data, span)
        latency = (time.monotonic() - t0) * 1000
        rest_result["latency_ms"] = round(latency, 2)
        span.set_attribute("ovms.latency_ms", latency)
        span.set_attribute("ovms.transport", "rest")
        return rest_result


# ---------------------------------------------------------------------------
# gRPC inference (primary)
# ---------------------------------------------------------------------------
async def _try_grpc_inference(model_name: str, input_data: dict, span: Any) -> dict | None:
    """Attempt gRPC predict call.  Returns None if gRPC is unavailable."""
    svc_grpc, predict_pb2, make_tensor_proto = _get_predict_service_stub()
    if svc_grpc is None:
        logger.debug("gRPC stubs not available -- will use REST fallback")
        return None

    channel = await _ensure_grpc_channel()
    if channel is None:
        return None

    try:
        stub = svc_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name

        # Populate inputs
        if make_tensor_proto is not None:
            for key, val in input_data.items():
                request.inputs[key].CopyFrom(make_tensor_proto(val))
        else:
            # Without make_tensor_proto we cannot build TensorProtos;
            # fall back to REST.
            return None

        response = await stub.Predict(request, timeout=GRPC_TIMEOUT_S)

        # Extract outputs
        outputs = {}
        for key in response.outputs:
            tensor = response.outputs[key]
            vals = list(tensor.float_val) or list(tensor.int_val) or list(tensor.double_val)
            outputs[key] = vals

        span.set_attribute("ovms.result", "ok")
        return {"status": "ok", "model": model_name, "outputs": outputs}

    except grpc.aio.AioRpcError as exc:
        code = exc.code()
        if code == grpc.StatusCode.NOT_FOUND:
            return {
                "status": "model_not_found",
                "message": f"gRPC: model '{model_name}' not found",
            }
        logger.warning("gRPC inference failed (code=%s): %s", code, exc.details())
        return None  # trigger REST fallback
    except Exception as exc:
        logger.warning("gRPC inference error: %s", exc)
        return None  # trigger REST fallback


# ---------------------------------------------------------------------------
# REST inference (fallback)
# ---------------------------------------------------------------------------
async def _try_rest_inference(model_name: str, input_data: dict, span: Any) -> dict:
    """REST predict call via OVMS TensorFlow Serving compatible API."""
    # OVMS supports TFS REST API: POST /v1/models/{name}:predict
    url = f"{OVMS_REST_BASE}/v1/models/{model_name}:predict"
    payload = {"inputs": input_data}

    try:
        async with httpx.AsyncClient(timeout=REST_TIMEOUT_S) as client:
            resp = await client.post(url, json=payload)

            if resp.status_code == 200:
                body = resp.json()
                span.set_attribute("ovms.result", "ok")
                return {
                    "status": "ok",
                    "model": model_name,
                    "outputs": body.get("outputs", body),
                }
            elif resp.status_code == 404:
                span.set_attribute("ovms.result", "model_not_found")
                return {
                    "status": "model_not_found",
                    "message": f"REST 404: model '{model_name}' not served",
                }
            else:
                error_text = resp.text[:500]
                span.set_attribute("ovms.result", "error")
                logger.warning(
                    "OVMS REST returned %d for model %s: %s",
                    resp.status_code,
                    model_name,
                    error_text,
                )
                return {
                    "status": "error",
                    "message": f"OVMS returned HTTP {resp.status_code}: {error_text}",
                }
    except httpx.ConnectError:
        span.set_attribute("ovms.result", "unreachable")
        return {
            "status": "error",
            "message": f"OVMS REST unreachable at {OVMS_REST_BASE}",
        }
    except Exception as exc:
        span.set_attribute("ovms.result", "error")
        return {"status": "error", "message": str(exc)}
