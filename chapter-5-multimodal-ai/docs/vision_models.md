# Vision Models Guide

## CLIP (Contrastive Language-Image Pre-training)

### Overview
CLIP learns visual concepts from natural language supervision, enabling zero-shot classification and semantic search.

### Key Features
- Zero-shot classification without training data
- Semantic image search using text queries
- Joint image-text embeddings

### Usage
```python
from vision.clip_integration import CLIPIntegration

clip = CLIPIntegration(model_name="ViT-B/32")

# Zero-shot classification
results = clip.zero_shot_classification(
    "image.jpg",
    ["dog", "cat", "bird"]
)

# Semantic search
matches = clip.semantic_image_search(
    image_paths=["img1.jpg", "img2.jpg"],
    query="sunset over ocean",
    top_k=5
)
```

### Performance
- Model: ViT-B/32
- Embedding size: 512 dimensions
- Inference time: ~50ms per image (GPU)

## BLIP/BLIP-2

### Overview
BLIP (Bootstrapping Language-Image Pre-training) generates captions and answers questions about images.

### Key Features
- Image captioning
- Visual question answering
- Conditional text generation

### BLIP-2 Improvements
- Q-Former architecture bridges frozen vision and language models
- Better efficiency and performance
- Supports larger language models

### Usage
```python
from vision.blip_integration import BLIPIntegration, BLIP2Integration

# BLIP captioning
blip = BLIPIntegration()
caption = blip.generate_caption("image.jpg")

# BLIP-2 VQA
blip2 = BLIP2Integration()
answer = blip2.answer_question(
    "image.jpg",
    "What color is the car?"
)
```

### Performance
- BLIP-base: ~100ms per image (GPU)
- BLIP-2: ~150ms per image (GPU)

## Grounding DINO

### Overview
Open-vocabulary object detection that can detect objects based on text descriptions.

### Key Features
- Zero-shot object detection
- Text-conditioned detection
- Bounding box generation

### Usage
```python
from vision.grounding_dino import GroundingDINOIntegration

dino = GroundingDINOIntegration()

detections = dino.detect_objects(
    "image.jpg",
    text_prompt="person . car . dog",
    box_threshold=0.35
)

# Save annotated image
dino.detect_and_annotate(
    "image.jpg",
    "person . car",
    "output.jpg"
)
```

### Performance
- Inference time: ~200ms per image (GPU)
- Accurate for common objects
- May struggle with uncommon objects

## Model Selection Guide

| Task | Recommended Model | Reason |
|------|------------------|---------|
| Zero-shot classification | CLIP | Fast, accurate |
| Image captioning | BLIP-2 | Best quality |
| Visual Q&A | BLIP-2 or Claude Vision | Flexible, accurate |
| Object detection | Grounding DINO | Open vocabulary |
| General image analysis | Claude Vision | Most versatile |

## Optimization Tips

1. **Batch Processing**: Process multiple images together
2. **Caching**: Cache embeddings for frequently used images
3. **Model Selection**: Use smaller models when possible
4. **GPU Utilization**: Ensure CUDA is properly configured
5. **Preprocessing**: Resize images before inference