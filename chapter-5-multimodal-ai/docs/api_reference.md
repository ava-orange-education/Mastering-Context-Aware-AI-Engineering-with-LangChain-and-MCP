# API Reference

## Core Classes

### MultiModalAgent

Base agent for multi-modal processing.
```python
class MultiModalAgent:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022")
    
    def encode_image(self, image_path: str) -> tuple[str, str]
    
    def analyze_image(self, image_path: str, prompt: str) -> str
    
    def analyze_multiple_images(self, image_paths: List[str], prompt: str) -> str
```

## Personal Assistant

### MultiModalPersonalAssistant

Main assistant class with all capabilities.
```python
class MultiModalPersonalAssistant:
    def __init__(self, api_key: str, enable_cache: bool = True)
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]
    
    def analyze_document(self, document_path: str, questions: List[str]) -> Dict[str, Any]
    
    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]
    
    def analyze_image(self, image_path: str, query: str) -> Dict[str, Any]
    
    def search_images(self, image_paths: List[str], query: str, top_k: int = 5) -> Dict[str, Any]
    
    def get_stats(self) -> Dict[str, Any]
```

## Vision Components

### CLIPIntegration
```python
class CLIPIntegration:
    def __init__(self, model_name: str = "ViT-B/32")
    
    def encode_image(self, image_path: str) -> np.ndarray
    
    def encode_text(self, text: str) -> np.ndarray
    
    def zero_shot_classification(self, image_path: str, candidate_labels: List[str]) -> List[Tuple[str, float]]
    
    def semantic_image_search(self, image_paths: List[str], query: str, top_k: int = 5) -> List[Tuple[str, float]]
```

### BLIPIntegration
```python
class BLIPIntegration:
    def __init__(self, model_type: str = "blip-base")
    
    def generate_caption(self, image_path: str, max_length: int = 50, num_beams: int = 5) -> str
    
    def conditional_caption(self, image_path: str, prompt: str, max_length: int = 50) -> str
```

### BLIP2Integration
```python
class BLIP2Integration:
    def __init__(self, model_type: str = "blip2-opt-2.7b")
    
    def answer_question(self, image_path: str, question: str, max_length: int = 50) -> str
    
    def batch_vqa(self, image_path: str, questions: List[str]) -> List[str]
```

## Audio Components

### WhisperIntegration
```python
class WhisperIntegration:
    def __init__(self, model_size: str = "base")
    
    def transcribe(self, audio_path: str, language: Optional[str] = None, task: str = "transcribe") -> Dict[str, Any]
    
    def transcribe_with_timestamps(self, audio_path: str) -> List[Dict[str, Any]]
    
    def detect_language(self, audio_path: str) -> Dict[str, float]
```

## MCP Agents

### MultiModalOrchestrator
```python
class MultiModalOrchestrator:
    def __init__(self, api_key: str)
    
    def route_request(self, request: Dict[str, Any]) -> str
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]
    
    def cross_modal_reasoning(self, image_path: Optional[str] = None, audio_path: Optional[str] = None, document_path: Optional[str] = None, query: str = "") -> Dict[str, Any]
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]
```

## Utilities

### InputValidator
```python
class InputValidator:
    @staticmethod
    def validate_image(image_path: str) -> Dict[str, Any]
    
    @staticmethod
    def validate_audio(audio_path: str) -> Dict[str, Any]
    
    @staticmethod
    def validate_document(document_path: str) -> Dict[str, Any]
    
    @staticmethod
    def validate_request(request: Dict[str, Any]) -> Dict[str, Any]
```

### MultiModalCache
```python
class MultiModalCache:
    def __init__(self, cache_dir: str = "./cache", ttl_seconds: int = 3600)
    
    def get(self, request: Dict[str, Any]) -> Optional[Any]
    
    def set(self, request: Dict[str, Any], response: Any)
    
    def clear(self)
    
    def get_stats(self) -> Dict[str, Any]
```

## Error Handling

### RetryHandler
```python
class RetryHandler:
    @staticmethod
    def exponential_backoff(func: Callable, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0, exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Callable
```

## Cost Tracking

### CostTracker
```python
class CostTracker:
    def __init__(self)
    
    def record_llm_call(self, model: str, input_tokens: int, output_tokens: int, task: str = 'general')
    
    def record_whisper_call(self, audio_duration_seconds: float)
    
    def record_image_processing(self, num_images: int, model: str = 'clip')
    
    def get_summary(self) -> Dict[str, Any]
    
    def check_budget(self, budget_limit: float) -> Dict[str, Any]
```