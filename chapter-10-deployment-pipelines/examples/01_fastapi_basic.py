"""
Example 1: Basic FastAPI Usage
Demonstrates basic API endpoints and request/response handling
"""

import requests
import json
from pprint import pprint

# API base URL
BASE_URL = "http://localhost:8000"


def test_root_endpoint():
    """Test root endpoint"""
    print("\n" + "="*70)
    print("Testing Root Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    pprint(response.json())


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("Testing Health Check")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    pprint(response.json())


def test_detailed_health_check():
    """Test detailed health check"""
    print("\n" + "="*70)
    print("Testing Detailed Health Check")
    print("="*70)
    
    payload = {
        "check_vector_store": True,
        "check_llm": True,
        "check_agents": True
    }
    
    response = requests.post(
        f"{BASE_URL}/health/detailed",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    pprint(response.json())


def test_query_endpoint():
    """Test query processing endpoint"""
    print("\n" + "="*70)
    print("Testing Query Endpoint")
    print("="*70)
    
    payload = {
        "query": "What is machine learning?",
        "top_k": 5,
        "agent_type": "multi_agent"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/agents/query",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    pprint(response.json())


def test_system_metrics():
    """Test system metrics endpoint"""
    print("\n" + "="*70)
    print("Testing System Metrics")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/metrics/system")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    pprint(response.json())


def test_validation_error():
    """Test request validation"""
    print("\n" + "="*70)
    print("Testing Validation Error Handling")
    print("="*70)
    
    # Send invalid request (empty query)
    payload = {
        "query": "",
        "top_k": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/agents/query",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    pprint(response.json())


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FastAPI Basic Usage Examples")
    print("="*70)
    print("\nMake sure the API is running on http://localhost:8000")
    print("Start with: uvicorn api.main:app --reload")
    
    try:
        # Test basic endpoints
        test_root_endpoint()
        test_health_check()
        test_detailed_health_check()
        test_system_metrics()
        
        # Test query endpoint
        test_query_endpoint()
        
        # Test error handling
        test_validation_error()
        
        print("\n" + "="*70)
        print("All tests completed!")
        print("="*70)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the API is running on http://localhost:8000")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")