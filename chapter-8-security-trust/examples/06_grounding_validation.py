"""
Example 6: Grounding and Citation Validation
Demonstrates citation management and fact checking.
"""

import sys
sys.path.append('..')

from grounding.citation_manager import CitationManager
from grounding.fact_checker import FactChecker
from grounding.confidence_scorer import ConfidenceScorer
from anthropic import Anthropic
import os


def main():
    print("=" * 70)
    print("  GROUNDING AND CITATION VALIDATION EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize components
    citation_mgr = CitationManager()
    llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    fact_checker = FactChecker(llm)
    confidence_scorer = ConfidenceScorer()
    
    print("Phase 1: Citation Management")
    print("-" * 70)
    
    # Add citations
    cite1 = citation_mgr.add_citation(
        source_title="Q4 2023 Financial Report",
        source_url="https://example.com/reports/q4-2023",
        excerpt="Revenue reached $500M in Q4, up 20% from Q3",
        source_type="report"
    )
    
    cite2 = citation_mgr.add_citation(
        source_title="Customer Satisfaction Survey 2023",
        source_url="https://example.com/surveys/satisfaction-2023",
        excerpt="Customer satisfaction score improved to 4.5/5 stars",
        source_type="survey"
    )
    
    print(f"✓ Added {len(citation_mgr.citations)} citations\n")
    
    # Create response with citations
    response_text = "The company performed well in Q4 2023, with revenue reaching $500M. Customer satisfaction also improved to 4.5 out of 5 stars."
    
    source_docs = [
        {
            'title': 'Q4 2023 Financial Report',
            'url': 'https://example.com/reports/q4-2023',
            'content': 'Revenue reached $500M in Q4, up 20% from Q3. Profit margins improved.',
            'type': 'report'
        },
        {
            'title': 'Customer Satisfaction Survey 2023',
            'url': 'https://example.com/surveys/satisfaction-2023',
            'content': 'Customer satisfaction score improved to 4.5/5 stars. Response rate: 85%.',
            'type': 'survey'
        }
    ]
    
    cited_response = citation_mgr.create_cited_response(response_text, source_docs)
    
    print("Response with citations:")
    formatted = citation_mgr.format_response_with_citations(cited_response)
    print(formatted)
    
    print(f"\nGrounding score: {cited_response.grounding_score:.2f}")
    
    print("\n\nPhase 2: Citation Verification")
    print("-" * 70)
    
    verification = citation_mgr.verify_citations(response_text, cited_response.citations)
    
    print(f"Well grounded: {verification['well_grounded']}")
    print(f"Grounding rate: {verification['grounding_rate']:.2%}")
    print(f"Grounded sentences: {verification['grounded_sentences']}/{verification['total_sentences']}")
    
    print("\n\nPhase 3: Fact Checking")
    print("-" * 70)
    
    # Test fact checking
    claims = [
        "Revenue reached $500M in Q4 2023",
        "Customer satisfaction improved to 4.5/5",
        "The company launched 10 new products"  # Not in sources
    ]
    
    sources = [doc['content'] for doc in source_docs]
    
    print("Checking claims against sources:\n")
    
    for claim in claims:
        print(f"Claim: \"{claim}\"")
        result = fact_checker.check_fact(claim, sources)
        
        print(f"  Supported: {result['supported']}")
        print(f"  Verdict: {result['verdict']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Explanation: {result['explanation']}\n")
    
    print("\nPhase 4: Confidence Scoring")
    print("-" * 70)
    
    # Score different responses
    test_responses = [
        {
            'text': "Revenue definitely reached $500M with absolutely guaranteed growth.",
            'citations': 1,
            'grounding': 0.6,
            'desc': 'Overconfident language'
        },
        {
            'text': "Revenue appears to have reached approximately $500M, suggesting growth.",
            'citations': 2,
            'grounding': 0.9,
            'desc': 'Hedged language, well-cited'
        },
        {
            'text': "Based on the Q4 report, revenue was $500M.",
            'citations': 1,
            'grounding': 1.0,
            'desc': 'Direct citation'
        }
    ]
    
    for test in test_responses:
        print(f"\n{test['desc']}:")
        print(f"  Response: \"{test['text'][:50]}...\"")
        
        score = confidence_scorer.score_response(
            response=test['text'],
            citations_count=test['citations'],
            grounding_score=test['grounding']
        )
        
        print(f"  Confidence: {score['confidence_score']:.2f} ({score['confidence_level']})")
        print(f"  Grounding: {score['grounding_score']:.2f}")
        print(f"  Citations: {score['citations_count']}")
        print(f"  Hedging words: {score['hedging_count']}")
        print(f"  Overconfident words: {score['overconfident_count']}")
    
    print("\n" + "=" * 70)
    print("  GROUNDING VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()