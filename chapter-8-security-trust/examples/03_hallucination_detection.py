"""
Example 3: Hallucination Detection
Demonstrates detecting potential hallucinations in AI responses.
"""

import sys
sys.path.append('..')

from security.hallucination_detector import HallucinationDetector
from anthropic import Anthropic
import os


def main():
    print("=" * 70)
    print("  HALLUCINATION DETECTION EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize detector
    llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    detector = HallucinationDetector(llm)
    
    print("Phase 1: Pattern-Based Detection")
    print("-" * 70)
    
    # Test responses with suspicious patterns
    test_cases = [
        {
            'response': "Studies definitely show that AI will absolutely replace all jobs by 2025.",
            'context': None,
            'expected': 'overconfident language'
        },
        {
            'response': "Research indicates that the market grew 47.3% in Q3 2023.",
            'context': None,
            'expected': 'specific numbers without citation'
        },
        {
            'response': "According to the 2023 report, revenue increased moderately.",
            'context': "2023 Annual Report: Revenue grew 15% year-over-year.",
            'expected': 'properly grounded'
        }
    ]
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\nTest Case {idx}: {test['expected']}")
        print(f"Response: \"{test['response'][:60]}...\"")
        
        result = detector.check_response(test['response'], test['context'])
        
        print(f"Hallucination detected: {result.is_hallucination}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reason: {result.reason}")
        print(f"Severity: {result.severity}")
    
    print("\n\nPhase 2: Context Grounding Check")
    print("-" * 70)
    
    context = """
    Company Q4 Report: Revenue reached $500M, up 20% from Q3.
    Customer satisfaction improved to 4.5/5 stars.
    New product launch scheduled for Q1 2024.
    """
    
    grounded_response = "Revenue was $500M in Q4, showing 20% growth."
    ungrounded_response = "Revenue exceeded $800M with 50% growth."
    
    print("\nGrounded response test:")
    result1 = detector.check_response(grounded_response, context)
    print(f"  Hallucination: {result1.is_hallucination}")
    print(f"  Reason: {result1.reason}")
    
    print("\nUngrounded response test:")
    result2 = detector.check_response(ungrounded_response, context)
    print(f"  Hallucination: {result2.is_hallucination}")
    print(f"  Reason: {result2.reason}")
    
    print("\n" + "=" * 70)
    print("  HALLUCINATION DETECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()