"""
Example 2: Image captioning with BLIP.
"""

import sys
sys.path.append('..')

from vision.blip_integration import BLIPIntegration, BLIP2Integration


def main():
    print("=== Example 2: Image Captioning ===\n")
    
    # Initialize BLIP
    print("Loading BLIP model...")
    blip = BLIPIntegration()
    
    # Example 1: Generate caption
    print("\n1. Generating caption for image...")
    caption = blip.generate_caption("../data/sample_images/scene.jpg")
    print(f"Caption: {caption}")
    
    # Example 2: Conditional caption
    print("\n2. Generating conditional caption...")
    caption = blip.conditional_caption(
        image_path="../data/sample_images/scene.jpg",
        prompt="a photo of"
    )
    print(f"Conditional Caption: {caption}")
    
    # Example 3: BLIP-2 Visual Question Answering
    print("\n3. Visual Question Answering with BLIP-2...")
    blip2 = BLIP2Integration()
    
    questions = [
        "What is the weather like in this image?",
        "How many people are visible?",
        "What time of day is it?"
    ]
    
    for question in questions:
        answer = blip2.answer_question(
            image_path="../data/sample_images/scene.jpg",
            question=question
        )
        print(f"Q: {question}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()