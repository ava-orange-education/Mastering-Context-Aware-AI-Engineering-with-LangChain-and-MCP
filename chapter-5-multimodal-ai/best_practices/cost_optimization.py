"""
Cost optimization strategies for multimodal AI.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CostTracker:
    """Track API costs and usage"""
    
    # Pricing (example rates - adjust based on actual pricing)
    PRICING = {
        'claude-3-5-sonnet-20241022': {
            'input': 0.003 / 1000,   # per token
            'output': 0.015 / 1000    # per token
        },
        'gpt-4-vision': {
            'input': 0.01 / 1000,
            'output': 0.03 / 1000
        },
        'whisper': {
            'per_minute': 0.006
        },
        'clip': {
            'per_image': 0.0001  # Local inference, minimal cost
        }
    }
    
    def __init__(self):
        self.costs = {
            'total': 0.0,
            'by_model': {},
            'by_task': {}
        }
        self.usage = {
            'api_calls': 0,
            'tokens_processed': 0,
            'images_processed': 0,
            'audio_minutes': 0
        }
    
    def record_llm_call(self, 
                       model: str,
                       input_tokens: int,
                       output_tokens: int,
                       task: str = 'general'):
        """
        Record LLM API call
        
        Args:
            model: Model name
            input_tokens: Input tokens
            output_tokens: Output tokens
            task: Task type
        """
        if model not in self.PRICING:
            logger.warning(f"Unknown model pricing: {model}")
            return
        
        pricing = self.PRICING[model]
        cost = (input_tokens * pricing['input'] + 
                output_tokens * pricing['output'])
        
        self.costs['total'] += cost
        self.costs['by_model'][model] = self.costs['by_model'].get(model, 0.0) + cost
        self.costs['by_task'][task] = self.costs['by_task'].get(task, 0.0) + cost
        
        self.usage['api_calls'] += 1
        self.usage['tokens_processed'] += input_tokens + output_tokens
    
    def record_whisper_call(self, audio_duration_seconds: float):
        """Record Whisper transcription"""
        minutes = audio_duration_seconds / 60
        cost = minutes * self.PRICING['whisper']['per_minute']
        
        self.costs['total'] += cost
        self.costs['by_model']['whisper'] = self.costs['by_model'].get('whisper', 0.0) + cost
        
        self.usage['audio_minutes'] += minutes
    
    def record_image_processing(self, num_images: int, model: str = 'clip'):
        """Record image processing"""
        if model in self.PRICING:
            cost = num_images * self.PRICING[model].get('per_image', 0.0)
            self.costs['total'] += cost
            self.costs['by_model'][model] = self.costs['by_model'].get(model, 0.0) + cost
        
        self.usage['images_processed'] += num_images
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        return {
            'total_cost': f"${self.costs['total']:.4f}",
            'costs_by_model': {
                model: f"${cost:.4f}"
                for model, cost in self.costs['by_model'].items()
            },
            'costs_by_task': {
                task: f"${cost:.4f}"
                for task, cost in self.costs['by_task'].items()
            },
            'usage': self.usage
        }
    
    def check_budget(self, budget_limit: float) -> Dict[str, Any]:
        """Check if within budget"""
        remaining = budget_limit - self.costs['total']
        percentage_used = (self.costs['total'] / budget_limit) * 100
        
        return {
            'budget_limit': budget_limit,
            'total_spent': self.costs['total'],
            'remaining': remaining,
            'percentage_used': percentage_used,
            'within_budget': self.costs['total'] <= budget_limit
        }


class CostOptimizedAssistant:
    """Assistant with cost optimization"""
    
    def __init__(self, api_key: str, budget_limit: Optional[float] = None):
        """
        Initialize cost-optimized assistant
        
        Args:
            api_key: API key
            budget_limit: Optional budget limit in USD
        """
        from personal_assistant import MultiModalPersonalAssistant
        
        self.assistant = MultiModalPersonalAssistant(api_key, enable_cache=True)
        self.cost_tracker = CostTracker()
        self.budget_limit = budget_limit
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request with cost tracking
        
        Args:
            request: Request dictionary
            
        Returns:
            Result with cost information
        """
        # Check budget before processing
        if self.budget_limit:
            budget_check = self.cost_tracker.check_budget(self.budget_limit)
            if not budget_check['within_budget']:
                return {
                    'success': False,
                    'error': 'Budget limit exceeded',
                    'budget_info': budget_check
                }
        
        # Process request
        result = self.assistant.process_request(request)
        
        # Track costs (simplified - in production, extract from API responses)
        task = request.get('task', 'general')
        
        if 'image' in task:
            self.cost_tracker.record_image_processing(1)
        
        if 'audio' in task:
            self.cost_tracker.record_whisper_call(60)  # Assume 1 minute
        
        # Add cost info to result
        result['cost_info'] = self.cost_tracker.get_summary()
        
        return result
    
    def optimize_image_batch(self, 
                           image_paths: list,
                           task: str,
                           batch_size: int = 5) -> list:
        """
        Process images in optimized batches
        
        Args:
            image_paths: List of image paths
            task: Task to perform
            batch_size: Batch size for processing
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            
            # Process batch together (more cost-effective)
            batch_result = self.assistant.process_request({
                'task': task,
                'image_paths': batch
            })
            
            results.append(batch_result)
        
        return results
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        summary = self.cost_tracker.get_summary()
        
        if self.budget_limit:
            summary['budget'] = self.cost_tracker.check_budget(self.budget_limit)
        
        return summary