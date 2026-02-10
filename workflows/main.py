import uvicorn
import logging
import sys
import os

# Ensure the parent directory is in the path for internal imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.a2a_server import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("antigravity")

def main():
    logger.info("=== ANTIGRAVITY NODE v13.0 STARTING ===")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

if __name__ == "__main__":
    main()
