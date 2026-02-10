from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root(): return {"status": "online"}
@app.get("/health")
def health(): return {"status": "healthy"}
