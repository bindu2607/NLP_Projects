#!/bin/bash

# Stop any running containers from previous deployments
docker-compose -f docker/docker-compose.yml down

# Build and start all services in the background
docker-compose -f docker/docker-compose.yml up --build -d

# Show the status of all containers
docker-compose -f docker/docker-compose.yml ps

echo "Deployment complete! Your API is running."
echo "Swagger docs: http://localhost:8000/docs"
echo "Prometheus metrics: http://localhost:9090"
echo "Grafana dashboards: http://localhost:3000"
