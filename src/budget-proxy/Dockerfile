FROM python:3.11-slim
RUN pip install --no-cache-dir fastapi uvicorn httpx
COPY src/budget-proxy/proxy.py /app/proxy.py
EXPOSE 4000
RUN useradd -r -u 1001 appuser
USER appuser
CMD ["python", "/app/proxy.py"]
