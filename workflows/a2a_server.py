from fastapi import FastAPI
import logging
logger = logging.getLogger("antigravity.a2a")
app = FastAPI(title="Antigravity Node v13.0", version="13.0.0")
@app.get("/")
async def root():
    return {"status": "online", "version": "13.0"}
@app.get("/health")
async def health():
    return {"status": "healthy"}
