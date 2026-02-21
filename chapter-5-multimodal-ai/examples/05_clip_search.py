"""
Example 5: Semantic image search with CLIP.
"""

import sys
sys.path.append('..')

from vision.clip_integration import CLIPIntegration
from utils.file_utils import FileUtils


def main():
    print("=== Example 5: Semantic Image Search ===\n")
    
    # Initialize CLIP
    print("Loading CLIP model...")
    clip = CLIPIntegration()
    
    # Get all images in directory
    image_dir = "../data/sample_images"
    image_paths = FileUtils.list_files(
        image_dir,
        extensions=['.jpg', '.jpeg', '.png']
    )
    
    print(f"Found {len(image_paths)} images\n")
    
    # Example 1: Zero-shot classification
    print("1. Zero-shot classification:")
    test_image = image_paths[0]
    labels = ["dog", "cat", "car", "building", "nature", "food"]
    
    results = clip.zero_shot_classification(test_image, labels)
    
    print(f"Image: {test_image}")
    print("Classification results:")
    for label, prob in results[:3]:
        print(f"  {label}: {prob:.2%}")
    print()
    
    # Example 2: Semantic search
    print("2. Semantic image search:")
    
    search_queries = [
        "a photo of a sunset",
        "people smiling",
        "urban architecture"
    ]
    
    for query in search_queries:
        print(f"\nQuery: '{query}'")
        results = clip.semantic_image_search(image_paths, query, top_k=3)
        
        print("Top 3 results:")
        for i, (image_path, score) in enumerate(results, 1):
            print(f"  {i}. {image_path} (score: {score:.3f})")


if __name__ == "__main__":
    main()