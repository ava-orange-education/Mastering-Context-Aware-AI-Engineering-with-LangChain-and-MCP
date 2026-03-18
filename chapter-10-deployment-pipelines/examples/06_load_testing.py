"""
Example 6: Load Testing
Demonstrates load testing the API with concurrent requests
"""

import asyncio
import aiohttp
import time
from typing import List, Dict
import statistics


BASE_URL = "http://localhost:8000"


async def send_query(session: aiohttp.ClientSession, query: str, query_id: int) -> Dict:
    """Send a single query request"""
    payload = {
        "query": query,
        "top_k": 5,
        "agent_type": "multi_agent"
    }
    
    start_time = time.time()
    
    try:
        async with session.post(
            f"{BASE_URL}/api/v1/agents/query",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            duration = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                return {
                    "query_id": query_id,
                    "status": "success",
                    "duration": duration,
                    "response_time_ms": data.get("processing_time_ms", 0)
                }
            else:
                return {
                    "query_id": query_id,
                    "status": "error",
                    "duration": duration,
                    "error": f"HTTP {response.status}"
                }
    
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        return {
            "query_id": query_id,
            "status": "timeout",
            "duration": duration
        }
    
    except Exception as e:
        duration = time.time() - start_time
        return {
            "query_id": query_id,
            "status": "error",
            "duration": duration,
            "error": str(e)
        }


async def load_test(num_requests: int, concurrency: int):
    """Run load test with specified number of requests and concurrency"""
    
    print(f"\n{'='*70}")
    print(f"Load Test: {num_requests} requests, {concurrency} concurrent")
    print(f"{'='*70}\n")
    
    # Sample queries
    queries = [
        "What is machine learning?",
        "Explain deep learning",
        "What are neural networks?",
        "How does natural language processing work?",
        "What is computer vision?",
        "Explain reinforcement learning",
        "What is supervised learning?",
        "How do transformers work?",
        "What is transfer learning?",
        "Explain gradient descent"
    ]
    
    # Create query list
    test_queries = [queries[i % len(queries)] for i in range(num_requests)]
    
    # Run requests
    start_time = time.time()
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Create batches for concurrency control
        for i in range(0, num_requests, concurrency):
            batch = test_queries[i:i + concurrency]
            batch_ids = list(range(i, i + len(batch)))
            
            # Send batch concurrently
            tasks = [
                send_query(session, query, query_id)
                for query, query_id in zip(batch, batch_ids)
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Progress update
            completed = min(i + concurrency, num_requests)
            print(f"Completed: {completed}/{num_requests}")
    
    total_duration = time.time() - start_time
    
    # Analyze results
    analyze_results(results, total_duration, num_requests)


def analyze_results(results: List[Dict], total_duration: float, num_requests: int):
    """Analyze and display test results"""
    
    print(f"\n{'='*70}")
    print("Load Test Results")
    print(f"{'='*70}\n")
    
    # Count by status
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Calculate statistics
    successful = [r for r in results if r["status"] == "success"]
    
    if successful:
        durations = [r["duration"] for r in successful]
        response_times = [r["response_time_ms"] for r in successful]
        
        print("Overall Metrics:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Total Duration: {total_duration:.2f}s")
        print(f"  Requests/Second: {num_requests / total_duration:.2f}")
        print()
        
        print("Status Distribution:")
        for status, count in status_counts.items():
            percentage = (count / num_requests) * 100
            print(f"  {status.capitalize()}: {count} ({percentage:.1f}%)")
        print()
        
        print("Latency Statistics (Network + Processing):")
        print(f"  Mean: {statistics.mean(durations):.3f}s")
        print(f"  Median: {statistics.median(durations):.3f}s")
        print(f"  Min: {min(durations):.3f}s")
        print(f"  Max: {max(durations):.3f}s")
        print(f"  Stdev: {statistics.stdev(durations):.3f}s")
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        p50_idx = int(len(sorted_durations) * 0.50)
        p95_idx = int(len(sorted_durations) * 0.95)
        p99_idx = int(len(sorted_durations) * 0.99)
        
        print(f"  P50: {sorted_durations[p50_idx]:.3f}s")
        print(f"  P95: {sorted_durations[p95_idx]:.3f}s")
        print(f"  P99: {sorted_durations[p99_idx]:.3f}s")
        print()
        
        print("Processing Time Statistics (Server-side):")
        print(f"  Mean: {statistics.mean(response_times):.2f}ms")
        print(f"  Median: {statistics.median(response_times):.2f}ms")
        print(f"  Min: {min(response_times):.2f}ms")
        print(f"  Max: {max(response_times):.2f}ms")
    
    else:
        print("❌ No successful requests")
        print("\nErrors:")
        for result in results:
            if result["status"] == "error":
                print(f"  Query {result['query_id']}: {result.get('error', 'Unknown')}")


async def health_check():
    """Check if API is available"""
    print("Checking API availability...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print("✅ API is available\n")
                    return True
                else:
                    print(f"❌ API returned status {response.status}\n")
                    return False
    
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}\n")
        print(f"Make sure API is running at {BASE_URL}")
        return False


async def main():
    """Main function"""
    print("\n" + "="*70)
    print("API Load Testing")
    print("="*70)
    
    # Check API availability
    if not await health_check():
        return
    
    # Test scenarios
    print("Available test scenarios:")
    print("1. Light load (10 requests, concurrency 2)")
    print("2. Medium load (50 requests, concurrency 5)")
    print("3. Heavy load (100 requests, concurrency 10)")
    print("4. Stress test (200 requests, concurrency 20)")
    print("5. Custom")
    print("0. Exit")
    
    choice = input("\nSelect scenario (0-5): ")
    
    if choice == "1":
        await load_test(num_requests=10, concurrency=2)
    elif choice == "2":
        await load_test(num_requests=50, concurrency=5)
    elif choice == "3":
        await load_test(num_requests=100, concurrency=10)
    elif choice == "4":
        await load_test(num_requests=200, concurrency=20)
    elif choice == "5":
        num_requests = int(input("Number of requests: "))
        concurrency = int(input("Concurrency level: "))
        await load_test(num_requests=num_requests, concurrency=concurrency)
    elif choice == "0":
        print("Exiting...")
        return
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())