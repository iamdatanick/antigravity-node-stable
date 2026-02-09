@echo off
echo ============================================
echo   ANTIGRAVITY NODE v13.0 - STARTUP
echo ============================================
echo.

cd /d "C:\Users\NickV\OneDrive\Desktop\Antigravity-Node"

echo [1/4] Checking Python version...
python --version

echo.
echo [2/4] Checking Docker Desktop...
docker --version
if %errorlevel% neq 0 (
    echo ERROR: Docker not found. Please start Docker Desktop.
    pause
    exit /b 1
)

echo.
echo [3/4] Starting infrastructure services...
docker compose up -d postgres seaweedfs nats etcd

echo.
echo [4/4] Waiting for services to be ready (30 seconds)...
timeout /t 30 /nobreak

echo.
echo Starting Antigravity Node API...
echo ============================================
echo   API will be available at:
echo   - HTTP: http://localhost:8080
echo   - gRPC: localhost:8081
echo   - Health: http://localhost:8080/health
echo   - A2A Discovery: http://localhost:8080/.well-known/agent.json
echo ============================================
echo.

python workflows/main.py

pause
