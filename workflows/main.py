import uvicorn
import logging
import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.a2a_server import app

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
