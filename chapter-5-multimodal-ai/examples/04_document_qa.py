04_document_qa"""
Example 4: Document question answering.
"""

import sys
sys.path.append('..')

from llm_integration.claude_multimodal import ClaudeMultimodal


def main():
    print("=== Example 4: Document Question Answering ===\n")
    
    # Initialize Claude
    claude = ClaudeMultimodal(api_key="your-api-key-here")
    
    # Document to analyze
    document_path = "../data/sample_documents/contract.pdf"
    
    # Questions to answer
    questions = [
        "What is the contract duration?",
        "What are the payment terms?",
        "Who are the parties involved?",
        "What are the termination conditions?",
        "Are there any penalties mentioned?"
    ]
    
    print(f"Analyzing document: {document_path}\n")
    
    # Ask questions
    answers = claude.document_analysis(document_path, questions)
    
    # Display results
    for question, answer in zip(questions, answers):
        print(f"Q: {question}")
        print(f"A: {answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()