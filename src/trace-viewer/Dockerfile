FROM python:3.11-slim
RUN pip install --no-cache-dir streamlit pymysql pandas plotly
COPY trace_viewer.py /app/trace_viewer.py
EXPOSE 8501
CMD ["streamlit", "run", "/app/trace_viewer.py", "--server.port=8501", "--server.address=0.0.0.0"]
