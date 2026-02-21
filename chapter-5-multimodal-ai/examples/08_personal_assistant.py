"""
Example 8: Complete personal assistant usage.
"""

import sys
sys.path.append('..')

from personal_assistant.assistant import MultiModalPersonalAssistant


def main():
    print("=== Example 8: Multi-Modal Personal Assistant ===\n")
    
    # Initialize assistant
    assistant = MultiModalPersonalAssistant(
        api_key="your-api-key-here",
        enable_cache=True
    )
    
    # Example 1: Receipt analysis
    print("1. Receipt Analysis")
    result = assistant.process_request({
        'capability': 'receipt_analysis',
        'image_path': '../data/sample_images/receipt.jpg'
    })
    
    if result['success']:
        print("Receipt Information:")
        print(result['result']['extracted_info'])
    print()
    
    # Example 2: Meeting transcription
    print("2. Meeting Transcription and Summary")
    result = assistant.process_request({
        'capability': 'meeting_transcription',
        'audio_path': '../data/sample_audio/meeting.mp3'
    })
    
    if result['success']:
        print("Transcription:", result['result']['transcription'][:200], "...")
        print("\nSummary:", result['result']['summary'])
    print()
    
    # Example 3: Document verification
    print("3. Document Verification")
    result = assistant.process_request({
        'capability': 'document_verification',
        'document_path': '../data/sample_documents/invoice.pdf',
        'reference_path': '../data/sample_documents/template.pdf'
    })
    
    if result['success']:
        print("Verification Result:")
        print(result['result']['verification'])
    print()
    
    # Example 4: Image search
    print("4. Semantic Image Search")
    result = assistant.search_images(
        image_paths=[
            '../data/sample_images/photo1.jpg',
            '../data/sample_images/photo2.jpg',
            '../data/sample_images/photo3.jpg'
        ],
        query="sunset over water",
        top_k=3
    )
    
    if result['success']:
        print("Search Results:")
        for i, (path, score) in enumerate(result['result']['results'], 1):
            print(f"  {i}. {path}: {score:.3f}")
    print()
    
    # Show statistics
    print("Assistant Statistics:")
    stats = assistant.get_stats()
    print(stats)


if __name__ == "__main__":
    main()