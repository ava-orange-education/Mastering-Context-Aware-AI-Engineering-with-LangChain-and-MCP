# Docker Deployment Guide

Guide to deploying the RAG Multi-Agent System using Docker.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM available
- Environment variables configured

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd chapter-10-deployment-pipelines
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required variables:
```bash
ANTHROPIC_API_KEY=your_key
PINECONE_API_KEY=your_key
JWT_SECRET_KEY=your_secret
```

### 3. Build Images
```bash
# Build all images
docker-compose -f deployment/docker/docker-compose.yaml build
```

### 4. Start Services
```bash
# Start in detached mode
docker-compose -f deployment/docker/docker-compose.yaml up -d

# View logs
docker-compose -f deployment/docker/docker-compose.yaml logs -f
```

### 5. Verify Deployment
```bash
# Check health
curl http://localhost:8000/health

# Check running containers
docker ps
```

## Architecture
```
┌─────────────────┐
│   API Service   │ :8000
│  (Main API)     │
└────────┬────────┘
         │
    ┌────┴────┬────────┬─────────┐
    │         │        │         │
┌───▼───┐ ┌──▼───┐ ┌──▼────┐ ┌──▼─────┐
│Retriev│ │Analys│ │Synthes│ │Vector  │
│al     │ │is    │ │is     │ │Store   │
│:8001  │ │:8002 │ │:8003  │ │(Cloud) │
└───────┘ └──────┘ └───────┘ └────────┘
```

## Building Images

### Main API Image
```bash
docker build \
  -f deployment/docker/Dockerfile \
  -t rag-api:latest \
  .
```

### Agent Images
```bash
# Retrieval agent
docker build \
  -f deployment/docker/Dockerfile.retrieval \
  -t rag-retrieval:latest \
  .

# Analysis agent
docker build \
  -f deployment/docker/Dockerfile.analysis \
  -t rag-analysis:latest \
  .

# Synthesis agent
docker build \
  -f deployment/docker/Dockerfile.synthesis \
  -t rag-synthesis:latest \
  .
```

## Running Containers

### Single Container
```bash
# Run main API
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  --env-file .env \
  rag-api:latest

# View logs
docker logs -f rag-api

# Stop container
docker stop rag-api
docker rm rag-api
```

### Multi-Container with Docker Compose
```bash
# Start all services
docker-compose -f deployment/docker/docker-compose.yaml up -d

# Stop all services
docker-compose -f deployment/docker/docker-compose.yaml down

# Restart specific service
docker-compose -f deployment/docker/docker-compose.yaml restart api

# Scale service
docker-compose -f deployment/docker/docker-compose.yaml up -d --scale api=3
```

## Configuration

### Environment Variables

Pass via `.env` file or command line:
```bash
docker run \
  -e ANTHROPIC_API_KEY=your_key \
  -e ENVIRONMENT=production \
  rag-api:latest
```

### Volumes

Mount configuration files:
```bash
docker run \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/logs:/app/logs \
  rag-api:latest
```

### Networks

Create custom network:
```bash
# Create network
docker network create rag-network

# Run containers on network
docker run --network rag-network rag-api:latest
```

## Image Optimization

### Multi-Stage Build

The Dockerfile uses multi-stage builds to minimize image size:
```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /build
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . .
CMD ["uvicorn", "api.main:app"]
```

### Size Comparison
```bash
# Check image sizes
docker images | grep rag

# Output:
# rag-api        latest  500MB
# rag-retrieval  latest  450MB
# rag-analysis   latest  450MB
# rag-synthesis  latest  450MB
```

## Health Checks

### Container Health Check

Built-in health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

### Check Status
```bash
# View health status
docker ps

# Output shows (healthy) or (unhealthy)
# CONTAINER ID  IMAGE      STATUS
# 12345...      rag-api    Up 5 min (healthy)
```

## Logging

### View Logs
```bash
# Follow logs
docker logs -f rag-api

# Last 100 lines
docker logs --tail 100 rag-api

# With timestamps
docker logs -t rag-api

# Compose logs
docker-compose logs -f api
```

### Log Configuration

Configure JSON logging:
```bash
docker run \
  -e LOG_FORMAT=json \
  -e LOG_LEVEL=INFO \
  rag-api:latest
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs rag-api

# Inspect container
docker inspect rag-api

# Check if port is in use
lsof -i :8000
```

### Connection Issues
```bash
# Test from inside container
docker exec -it rag-api curl http://localhost:8000/health

# Check network
docker network inspect bridge
```

### High Memory Usage
```bash
# Check resource usage
docker stats rag-api

# Set memory limit
docker run --memory=1g --memory-swap=1g rag-api:latest
```

### Can't Connect to Vector Store
```bash
# Check environment variables
docker exec rag-api env | grep PINECONE

# Test connectivity
docker exec rag-api curl -I https://api.pinecone.io
```

## Production Deployment

### Security

1. **Use secrets management**
```bash
   docker run --secret api_key rag-api:latest
```

2. **Run as non-root user**
```dockerfile
   USER appuser
```

3. **Read-only root filesystem**
```bash
   docker run --read-only rag-api:latest
```

### Resource Limits
```bash
docker run \
  --cpus=2 \
  --memory=2g \
  --memory-swap=2g \
  rag-api:latest
```

### Update Strategy
```bash
# Pull new image
docker pull rag-api:latest

# Stop old container
docker stop rag-api

# Start new container
docker run -d --name rag-api rag-api:latest

# Remove old container
docker rm old-rag-api
```

## Docker Compose Production
```yaml
version: '3.8'

services:
  api:
    image: registry.example.com/rag-api:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
      restart_policy:
        condition: on-failure
        max_attempts: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Backup and Restore

### Export Container
```bash
# Commit container to image
docker commit rag-api rag-api-backup:$(date +%Y%m%d)

# Save image
docker save rag-api-backup:20240315 | gzip > backup.tar.gz
```

### Import Image
```bash
# Load image
docker load < backup.tar.gz

# Run from backup
docker run rag-api-backup:20240315
```

## Cleanup
```bash
# Stop all containers
docker-compose down

# Remove images
docker rmi rag-api:latest rag-retrieval:latest

# Remove unused images
docker image prune -a

# Remove volumes
docker volume prune

# Complete cleanup
docker system prune -a --volumes
```