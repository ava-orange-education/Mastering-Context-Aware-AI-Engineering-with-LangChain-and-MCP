"""
Example 9: Cross-modal reasoning across multiple inputs.
"""

import sys
sys.path.append('..')

from mcp_agents.orchestrator import MultiModalOrchestrator


def main():
    print("=== Example 9: Cross-Modal Reasoning ===\n")
    
    # Initialize orchestrator
    orchestrator = MultiModalOrchestrator(api_key="your-api-key-here")
    
    # Scenario: Analyzing a product from multiple sources
    print("Scenario: Comprehensive Product Analysis\n")
    
    result = orchestrator.cross_modal_reasoning(
        image_path='../data/sample_images/product_photo.jpg',
        audio_path='../data/sample_audio/customer_review.mp3',
        document_path='../data/sample_documents/product_manual.pdf',
        query="""Analyze this product comprehensively:
        1. What does the product look like (from image)?
        2. What are customers saying about it (from audio)?
        3. What are its key features and specifications (from manual)?
        4. Based on all evidence, what's your overall assessment?"""
    )
    
    print("="*60)
    print("CROSS-MODAL ANALYSIS RESULTS")
    print("="*60 + "\n")
    
    # Show individual modality results
    print("Evidence from each modality:")
    print("-"*60)
    
    for modality, evidence in result['modality_results'].items():
        print(f"\n{modality.upper()}:")
        if isinstance(evidence, dict):
            for key, value in evidence.items():
                if key not in ['task', 'image_path', 'audio_path', 'document_path']:
                    print(f"  {key}: {str(value)[:200]}...")
        else:
            print(f"  {str(evidence)[:200]}...")
    
    # Show synthesized answer
    print("\n" + "="*60)
    print("SYNTHESIZED ANSWER")
    print("="*60 + "\n")
    print(result['synthesized_answer'])


if __name__ == "__main__":
    main()