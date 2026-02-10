import asyncio
import glob
import logging
import os
import sys
import threading
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("antigravity")

def main():
    logger.info("=== ANTIGRAVITY NODE v13.0 STARTING ===")
    from workflows.a2a_server import app
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()
