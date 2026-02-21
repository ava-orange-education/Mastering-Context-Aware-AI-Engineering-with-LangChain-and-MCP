"""
Example 7: Multi-agent coordination with MCP.
"""

import sys
sys.path.append('..')

from mcp_agents.orchestrator import MultiModalOrchestrator


def main():
    print("=== Example 7: Multi-Agent MCP System ===\n")
    
    # Initialize orchestrator
    orchestrator = MultiModalOrchestrator(api_key="your-api-key-here")
    
    # Show available capabilities
    print("Available capabilities:")
    capabilities = orchestrator.get_agent_capabilities()
    for agent_name, caps in capabilities.items():
        print(f"\n{agent_name.upper()} Agent:")
        for cap in caps:
            print(f"  - {cap}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 1: Vision task
    print("1. Vision Task: Image Classification")
    result = orchestrator.process_request({
        'task': 'image_classification',
        'image_path': '../data/sample_images/animal.jpg',
        'labels': ['dog', 'cat', 'bird', 'horse']
    })
    print(f"Result: {result}\n")
    
    # Example 2: Audio task
    print("2. Audio Task: Transcription")
    result = orchestrator.process_request({
        'task': 'audio_transcription',
        'audio_path': '../data/sample_audio/speech.mp3'
    })
    print(f"Transcription: {result['transcription']['text'][:200]}...\n")
    
    # Example 3: Document task
    print("3. Document Task: Text Extraction")
    result = orchestrator.process_request({
        'task': 'text_extraction',
        'document_path': '../data/sample_documents/report.pdf'
    })
    print(f"Extracted text length: {len(result['extracted_text'])} characters\n")
    
    # Example 4: Cross-modal reasoning
    print("4. Cross-Modal Reasoning")
    result = orchestrator.cross_modal_reasoning(
        image_path='../data/sample_images/product.jpg',
        audio_path='../data/sample_audio/review.mp3',
        document_path='../data/sample_documents/specs.pdf',
        query="Based on the visual appearance, audio review, and specifications, would you recommend this product?"
    )
    print(f"Synthesized Answer: {result['synthesized_answer']}")


if __name__ == "__main__":
    main()