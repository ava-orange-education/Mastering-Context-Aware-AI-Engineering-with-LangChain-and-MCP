"""
Example 6: Object detection with Grounding DINO.
"""

import sys
sys.path.append('..')

from vision.grounding_dino import GroundingDINOIntegration


def main():
    print("=== Example 6: Object Detection with Grounding DINO ===\n")
    
    # Initialize Grounding DINO
    print("Loading Grounding DINO model...")
    dino = GroundingDINOIntegration()
    
    # Example 1: Detect specific objects
    print("\n1. Detecting objects in image...")
    
    image_path = "../data/sample_images/street_scene.jpg"
    text_prompt = "person . car . bicycle . traffic light"
    
    detections = dino.detect_objects(
        image_path=image_path,
        text_prompt=text_prompt,
        box_threshold=0.35
    )
    
    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['label']}: {det['score']:.2f} at {det['bbox']}")
    
    # Example 2: Detect and annotate
    print("\n2. Detecting and saving annotated image...")
    
    output_path = "../data/sample_images/annotated_output.jpg"
    detections = dino.detect_and_annotate(
        image_path=image_path,
        text_prompt=text_prompt,
        output_path=output_path
    )
    
    print(f"Annotated image saved to: {output_path}")
    print(f"Detected {len(detections)} objects")


if __name__ == "__main__":
    main()