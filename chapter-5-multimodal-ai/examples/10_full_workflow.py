"""
Example 10: Complete end-to-end workflow.
"""

import sys
sys.path.append('..')

from personal_assistant.assistant import MultiModalPersonalAssistant
from best_practices.cost_optimization import CostOptimizedAssistant
from best_practices.monitoring import PerformanceMonitor


def main():
    print("=== Example 10: Complete End-to-End Workflow ===\n")
    
    # Initialize cost-optimized assistant with monitoring
    print("1. Initializing system with cost tracking and monitoring...")
    
    cost_assistant = CostOptimizedAssistant(
        api_key="your-api-key-here",
        budget_limit=10.0  # $10 budget
    )
    
    monitor = PerformanceMonitor(cost_assistant.assistant)
    
    # Workflow: Process a business meeting
    print("\n2. Processing business meeting workflow...\n")
    
    # Step 1: Transcribe meeting audio
    print("Step 1: Transcribing meeting...")
    result1 = monitor.monitored_request({
        'task': 'audio_transcription',
        'audio_path': '../data/sample_audio/business_meeting.mp3'
    })
    
    if result1['success']:
        print(f"✓ Transcribed ({result1['monitoring']['latency']:.2f}s)")
        transcription = result1['result']['transcription']['text']
    
    # Step 2: Extract action items from slides
    print("\nStep 2: Analyzing presentation slides...")
    result2 = monitor.monitored_request({
        'task': 'document_qa',
        'document_path': '../data/sample_documents/presentation.pdf',
        'questions': [
            'What are the key action items?',
            'What are the deadlines mentioned?',
            'Who is responsible for each task?'
        ]
    })
    
    if result2['success']:
        print(f"✓ Analyzed ({result2['monitoring']['latency']:.2f}s)")
    
    # Step 3: Analyze whiteboard photos
    print("\nStep 3: Analyzing whiteboard photos...")
    result3 = monitor.monitored_request({
        'task': 'visual_qa',
        'image_path': '../data/sample_images/whiteboard.jpg',
        'question': 'Extract all the notes and diagrams from this whiteboard'
    })
    
    if result3['success']:
        print(f"✓ Analyzed ({result3['monitoring']['latency']:.2f}s)")
    
    # Step 4: Generate comprehensive meeting summary
    print("\nStep 4: Generating comprehensive summary...")
    
    from anthropic import Anthropic
    client = Anthropic(api_key=cost_assistant.assistant.orchestrator.vision_agent.claude.client.api_key)
    
    summary_prompt = f"""Based on the following meeting information, create a comprehensive summary:

TRANSCRIPTION:
{transcription[:1000]}...

PRESENTATION ACTION ITEMS:
{result2['result']}

WHITEBOARD NOTES:
{result3['result']}

Please provide:
1. Meeting summary
2. Key decisions
3. Action items with owners and deadlines
4. Follow-up items"""
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    
    meeting_summary = response.content[0].text
    print("✓ Summary generated\n")
    
    # Display results
    print("="*60)
    print("MEETING SUMMARY")
    print("="*60)
    print(meeting_summary)
    
    # Show performance metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    perf_report = monitor.get_performance_report()
    print(f"\nTotal Requests: {perf_report['metrics']['total_requests']}")
    print(f"Success Rate: {perf_report['metrics']['success_rate']:.2%}")
    print(f"Average Latency: {perf_report['metrics']['latency']['mean']:.2f}s")
    print(f"P95 Latency: {perf_report['metrics']['latency']['p95']:.2f}s")
    
    # Show cost summary
    print("\n" + "="*60)
    print("COST SUMMARY")
    print("="*60)
    
    cost_summary = cost_assistant.get_cost_summary()
    print(f"\nTotal Cost: {cost_summary['total_cost']}")
    
    if 'budget' in cost_summary:
        budget_info = cost_summary['budget']
        print(f"Budget Used: {budget_info['percentage_used']:.1f}%")
        print(f"Remaining: ${budget_info['remaining']:.4f}")
    
    # Health check
    print("\n" + "="*60)
    print("SYSTEM HEALTH")
    print("="*60)
    
    health = monitor.check_health()
    print(f"\nStatus: {health['status'].upper()}")
    print(f"Success Rate: {health['success_rate']:.2%}")
    print(f"P95 Latency: {health['p95_latency']:.2f}s")


if __name__ == "__main__":
    main()