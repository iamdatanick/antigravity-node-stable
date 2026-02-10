import uvicorn
import logging
import sys
import os
sys.path.insert(0, os.getcwd())
from workflows.a2a_server import app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("antigravity")
def main():
    logger.info("=== ANTIGRAVITY NODE v13.0 STARTING ===")
    uvicorn.run(app, host="0.0.0.0", port=8080)
if __name__ == "__main__":
    main()
