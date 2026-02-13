"""Antigravity Node v14.1 — Dual-Protocol Entry Point.

Runs FastAPI (HTTP/A2A on port 8080) + gRPC (Intel SuperBuilder on port 8081)
+ God Mode loop (background iterations).
"""

import asyncio
import glob
import logging
import os
import sys
import threading

# Ensure /app is on Python path so `from workflows.X import Y` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

# --- Structured Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("antigravity")

# --- Configuration ---
MAX_ITERATIONS = int(os.environ.get("GOD_MODE_ITERATIONS", "50"))


# --- God Mode Loop (background) ---
async def check_dependencies(loop_id: int) -> str:
    """Check stack health via HTTP endpoints."""
    import aiohttp

    checks = {
        "etcd": (f"http://{os.environ.get('ETCD_HOST', 'etcd')}:{os.environ.get('ETCD_PORT', '2379')}/health", [200]),
        "ceph": (
            f"http://{os.environ.get('CEPH_HOST', 'ceph-demo')}:{os.environ.get('CEPH_PORT', '8000')}",
            [200, 403, 405],
        ),
        "ovms": (f"http://{os.environ.get('OVMS_HOST', 'ovms')}:9001/v2/health/live", [200]),
        "openbao": (f"{os.environ.get('OPENBAO_ADDR', 'http://openbao:8200')}/v1/sys/health", [200]),
    }
    results = {}
    async with aiohttp.ClientSession() as session:
        for name, (url, accept_codes) in checks.items():
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    results[name] = resp.status in accept_codes
            except Exception:
                results[name] = False

    healthy = all(results.values())
    status = "HEALTHY" if healthy else "DEGRADED"
    logger.info(f"[Loop {loop_id}] Stack health: {status} {results}")
    return f"STACK_{status}"


def ingest_context(loop_id: int) -> str:
    """Scan for canonical governance files in /app/context/ (Downloads mount)."""
    search_paths = ["/app/context", "/app/well-known"]
    all_files = []
    for path in search_paths:
        if os.path.isdir(path):
            all_files.extend(glob.glob(os.path.join(path, "*.csv")))
            all_files.extend(glob.glob(os.path.join(path, "*CANONICAL*.xlsx")))
            all_files.extend(glob.glob(os.path.join(path, "*.pdf")))
            all_files.extend(glob.glob(os.path.join(path, "*.json")))

    if not all_files:
        logger.warning(f"[Loop {loop_id}] No canonical context files found.")
        return "WARNING: No canonical context files found."

    logger.info(f"[Loop {loop_id}] Context loaded: {len(all_files)} governance file(s).")
    return f"CONTEXT_LOADED: {len(all_files)} files"


async def god_mode_loop():
    """Background God Mode loop — runs MAX_ITERATIONS health/ingest cycles."""
    logger.info(f"=== ANTIGRAVITY NODE v14.1 GOD MODE ({MAX_ITERATIONS} iterations) ===")

    for i in range(1, MAX_ITERATIONS + 1):
        logger.info(f"GOD MODE LOOP {i}/{MAX_ITERATIONS}")
        try:
            health = await check_dependencies(loop_id=i)
            context = ingest_context(loop_id=i)
            logger.info(f"LOOP {i}: health={health}, context={context}")
        except Exception as e:
            logger.error(f"LOOP {i} ERROR: {type(e).__name__}: {e}")

        # Wait between iterations (30s in production, shorter for early loops)
        delay = min(30, 5 + i * 2)
        await asyncio.sleep(delay)

    logger.info("=== GOD MODE COMPLETE ===")


def start_god_mode_background():
    """Start the God Mode loop in a background thread with its own event loop."""

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(god_mode_loop())
        except Exception as e:
            logger.error(f"God Mode crashed: {e}")
        finally:
            loop.close()

    thread = threading.Thread(target=_run, name="god-mode", daemon=True)
    thread.start()
    logger.info("God Mode loop started in background thread")


# --- Main Entry Point ---
def main():
    logger.info("=== ANTIGRAVITY NODE v14.1 STARTING ===")
    logger.info(f"God Mode iterations: {MAX_ITERATIONS}")

    # 0. Initialize OpenTelemetry tracing
    from workflows.telemetry import init_telemetry

    init_telemetry()
    logger.info("OpenTelemetry tracing initialized")

    # 1. Start gRPC server in background thread (port 8081)
    try:
        from workflows.grpc_server import serve_grpc

        grpc_thread = threading.Thread(target=serve_grpc, name="grpc-server", daemon=True)
        grpc_thread.start()
        logger.info("gRPC server started on port 8081")
    except Exception as e:
        logger.warning(f"gRPC server failed to start: {e}. Continuing without gRPC.", exc_info=True)

    # 2. Start God Mode loop in background
    start_god_mode_background()

    # 3. Instrument FastAPI app with OpenTelemetry
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    from workflows.a2a_server import app

    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumented with OpenTelemetry")

    # 4. Start FastAPI server (port 8080) — blocks
    logger.info("Starting FastAPI A2A server on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
