# API Guide

Complete guide to using the RAG Multi-Agent System API.

## Base URL
```
Production: https://api.rag-system.example.com
Development: http://localhost:8000
```

## Authentication

### Obtaining a Token
```bash
# Request token (placeholder - implement actual auth endpoint)
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

### Using the Token

Include the token in the Authorization header:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/v1/agents/query
```

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "components": {
    "api": {
      "status": "healthy"
    }
  }
}
```

### Detailed Health Check

**POST** `/health/detailed`

Get detailed health information about all components.

**Request:**
```json
{
  "check_vector_store": true,
  "check_llm": true,
  "check_agents": true
}
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600.5,
  "components": {
    "api": {"status": "healthy"},
    "vector_store": {"status": "healthy", "latency_ms": 45.2},
    "llm": {"status": "healthy", "latency_ms": 120.5},
    "agent_retrieval": {"status": "healthy"},
    "agent_analysis": {"status": "healthy"},
    "agent_synthesis": {"status": "healthy"}
  }
}
```

### Process Query

**POST** `/api/v1/agents/query`

Process a query through the multi-agent pipeline.

**Request:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "agent_type": "multi_agent",
  "metadata": {
    "user_id": "user123"
  }
}
```

**Parameters:**
- `query` (string, required): User query (1-1000 characters)
- `top_k` (integer, optional): Number of documents to retrieve (1-20, default: 5)
- `agent_type` (string, optional): Agent type - `multi_agent`, `retrieval`, `analysis`, `synthesis` (default: `multi_agent`)
- `metadata` (object, optional): Additional metadata

**Response:**
```json
{
  "status": "success",
  "query": "What is machine learning?",
  "response": "Machine learning is a subset of artificial intelligence...",
  "retrieved_documents": [
    {
      "content": "Document content...",
      "metadata": {
        "id": "doc123",
        "score": 0.95,
        "source": "textbook"
      }
    }
  ],
  "agent_type": "multi_agent",
  "processing_time_ms": 1234.5,
  "evaluation_scores": {
    "relevance": 0.92,
    "coherence": 0.88
  },
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-03-15T10:30:00Z"
}
```

### Get Query Status

**GET** `/api/v1/agents/query/{query_id}`

Get status of a previously submitted query.

**Response:**
```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": { ... }
}
```

### Batch Query Processing

**POST** `/api/v1/agents/query/batch`

Process multiple queries in batch (requires authentication).

**Request:**
```json
[
  {
    "query": "What is AI?",
    "top_k": 3
  },
  {
    "query": "Explain neural networks",
    "top_k": 5
  }
]
```

**Limits:** Maximum 10 queries per batch

**Response:**
```json
{
  "status": "completed",
  "total_queries": 2,
  "results": [ ... ]
}
```

### System Statistics

**GET** `/api/v1/admin/stats`

Get system statistics (requires authentication and admin role).

**Response:**
```json
{
  "total_queries": 10000,
  "average_response_time_ms": 250.5,
  "success_rate": 0.98,
  "queries_last_hour": 45,
  "queries_last_24h": 890,
  "agent_usage": {
    "multi_agent": 7500,
    "retrieval": 1500,
    "analysis": 500,
    "synthesis": 500
  }
}
```

### Vector Store Statistics

**GET** `/api/v1/admin/vector-store/stats`

Get vector store statistics (requires authentication).

**Response:**
```json
{
  "status": "success",
  "stats": {
    "total_vector_count": 100000,
    "dimension": 1536,
    "index_fullness": 0.45
  }
}
```

## Error Responses

All errors follow this format:
```json
{
  "status": "error",
  "error_type": "ValidationError",
  "error_message": "Request validation failed",
  "detail": {
    "errors": [...]
  },
  "timestamp": "2024-03-15T10:30:00Z"
}
```

### Common Error Codes

- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing or invalid token)
- `403` - Forbidden (insufficient permissions)
- `422` - Validation Error (request doesn't match schema)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error
- `503` - Service Unavailable (backend services down)

## Rate Limits

Default rate limits by endpoint:

- **Query endpoints**: 30 requests/minute
- **Upload endpoints**: 10 requests/minute
- **Health checks**: 100 requests/minute
- **Admin endpoints**: 60 requests/minute

Rate limit headers:
```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1615809600
```

## Examples

### cURL
```bash
# Basic query
curl -X POST http://localhost:8000/api/v1/agents/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5
  }'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/agents/query",
    json={
        "query": "What is machine learning?",
        "top_k": 5,
        "agent_type": "multi_agent"
    }
)

data = response.json()
print(data["response"])
```

### JavaScript
```javascript
fetch('http://localhost:8000/api/v1/agents/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'What is machine learning?',
    top_k: 5
  })
})
.then(response => response.json())
.then(data => console.log(data.response));
```

## Best Practices

1. **Always check health before deployment**
```bash
   curl http://localhost:8000/health
```

2. **Use appropriate agent types**
   - `multi_agent`: Full pipeline (retrieval → analysis → synthesis)
   - `retrieval`: Document retrieval only (faster, for search)
   - `synthesis`: Generation only (when you have context)

3. **Implement retry logic**
```python
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
   def query_with_retry(query):
       return requests.post(url, json={"query": query})
```

4. **Monitor rate limits**
   - Check `X-RateLimit-Remaining` header
   - Implement backoff when approaching limit

5. **Handle errors gracefully**
```python
   try:
       response = requests.post(url, json=payload)
       response.raise_for_status()
       return response.json()
   except requests.exceptions.HTTPError as e:
       if e.response.status_code == 429:
           # Handle rate limit
           time.sleep(60)
       else:
           # Handle other errors
           logger.error(f"Request failed: {e}")
```

## OpenAPI Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`