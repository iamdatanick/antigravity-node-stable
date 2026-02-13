FROM python:3.11-slim
RUN pip install --no-cache-dir fastmcp mcp
COPY server.py /app/server.py
EXPOSE 8000
RUN useradd -r -u 1001 appuser
USER appuser
CMD ["python", "/app/server.py"]
