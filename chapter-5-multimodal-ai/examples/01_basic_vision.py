"""
Example 1: Basic vision capabilities with Claude.
"""

import sys
sys.path.append('..')

from core.multimodal_agent import MultiModalAgent


def main():
    # Initialize agent
    agent = MultiModalAgent(api_key="your-api-key-here")
    
    print("=== Example 1: Basic Vision Analysis ===\n")
    
    # Example 1: Analyze single image
    print("1. Analyzing product image...")
    result = agent.analyze_image(
        image_path="../data/sample_images/product.jpg",
        prompt="Describe this product in detail, including its features and potential uses."
    )
    print(f"Result: {result}\n")
    
    # Example 2: Compare images
    print("2. Comparing before and after images...")
    result = agent.analyze_multiple_images(
        image_paths=[
            "../data/sample_images/before.jpg",
            "../data/sample_images/after.jpg"
        ],
        prompt="Compare these two images and describe the differences."
    )
    print(f"Result: {result}\n")
    
    # Example 3: Extract information from screenshot
    print("3. Extracting information from screenshot...")
    result = agent.analyze_image(
        image_path="../data/sample_images/dashboard.png",
        prompt="Extract all the metrics and values shown in this dashboard."
    )
    print(f"Result: {result}\n")


if __name__ == "__main__":
    main()