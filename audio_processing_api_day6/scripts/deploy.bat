@echo off
REM === Audio Processing API Deployment Script ===

REM Move to the project root if needed (uncomment if running from scripts/)
REM cd ..

REM Stop and remove any running containers
echo Stopping existing containers...
docker compose -f docker/docker-compose.yml down
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to stop containers. Please check Docker is running.
    exit /b 1
)

REM Build and start containers in detached mode
echo Building and starting containers...
docker compose -f docker/docker-compose.yml up --build -d
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to build or start containers. Please check your Docker setup.
    exit /b 1
)

REM Show container status
echo.
echo Container status:
docker compose -f docker/docker-compose.yml ps

echo.
echo Deployment complete! Your API is running.
echo Swagger docs: http://localhost:8000/docs
echo Prometheus:   http://localhost:9090
echo Grafana:      http://localhost:3000
echo Nginx:        http://localhost

REM Pause so you can see the output (optional, remove if running from VSCode terminal)
REM pause
