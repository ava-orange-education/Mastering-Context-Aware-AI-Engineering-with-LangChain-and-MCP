"""
Multi-agent orchestrator for coordinating specialized agents.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class MultiModalOrchestrator:
    """Orchestrates multiple specialized agents"""
    
    def __init__(self, api_key: str):
        from .vision_agent import VisionAgent
        from .audio_agent import AudioAgent
        from .document_agent import DocumentAgent
        
        self.vision_agent = VisionAgent(api_key)
        self.audio_agent = AudioAgent(api_key)
        self.document_agent = DocumentAgent(api_key)
        
        self.agents = {
            'vision': self.vision_agent,
            'audio': self.audio_agent,
            'document': self.document_agent
        }
        
        self.request_log = []
    
    def route_request(self, request: Dict[str, Any]) -> str:
        """
        Determine which agent should handle the request
        
        Args:
            request: Request dictionary
            
        Returns:
            Agent name
        """
        task = request.get('task', '').lower()
        
        # Vision tasks
        if any(keyword in task for keyword in ['image', 'visual', 'picture', 'photo', 'classification', 'caption', 'object']):
            return 'vision'
        
        # Audio tasks
        elif any(keyword in task for keyword in ['audio', 'voice', 'speech', 'transcription', 'sound']):
            return 'audio'
        
        # Document tasks
        elif any(keyword in task for keyword in ['document', 'pdf', 'text', 'extract', 'summarize']):
            return 'document'
        
        # Default to vision if unsure
        return 'vision'
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request by routing to appropriate agent
        
        Args:
            request: Request dictionary with 'task' and parameters
            
        Returns:
            Result from the agent
        """
        # Determine agent
        agent_name = request.get('agent')
        if not agent_name:
            agent_name = self.route_request(request)
        
        logger.info(f"Routing request to {agent_name} agent")
        
        # Get agent
        agent = self.agents.get(agent_name)
        if not agent:
            return {'error': f'Unknown agent: {agent_name}'}
        
        # Process request
        try:
            result = agent.handle_request(request)
            result['agent'] = agent_name
            
            # Log request
            self.request_log.append({
                'request': request,
                'agent': agent_name,
                'result': result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                'error': str(e),
                'agent': agent_name,
                'request': request
            }
    
    def process_multi_modal_request(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple related requests across different modalities
        
        Args:
            requests: List of request dictionaries
            
        Returns:
            Combined results
        """
        results = []
        
        for request in requests:
            result = self.process_request(request)
            results.append(result)
        
        return {
            'multi_modal': True,
            'num_requests': len(requests),
            'results': results
        }
    
    def cross_modal_reasoning(self, 
                             image_path: Optional[str] = None,
                             audio_path: Optional[str] = None,
                             document_path: Optional[str] = None,
                             query: str = "") -> Dict[str, Any]:
        """
        Perform cross-modal reasoning across multiple inputs
        
        Args:
            image_path: Optional image path
            audio_path: Optional audio path
            document_path: Optional document path
            query: Query about the inputs
            
        Returns:
            Integrated analysis
        """
        modality_results = {}
        
        # Process each modality
        if image_path:
            result = self.vision_agent.handle_request({
                'task': 'visual_qa',
                'image_path': image_path,
                'question': query
            })
            modality_results['vision'] = result
        
        if audio_path:
            result = self.audio_agent.handle_request({
                'task': 'audio_transcription',
                'audio_path': audio_path
            })
            modality_results['audio'] = result
        
        if document_path:
            result = self.document_agent.handle_request({
                'task': 'text_extraction',
                'document_path': document_path
            })
            modality_results['document'] = result
        
        # Combine evidence
        from anthropic import Anthropic
        client = Anthropic(api_key=self.vision_agent.claude.client.api_key)
        
        evidence_parts = []
        for modality, result in modality_results.items():
            evidence_parts.append(f"{modality.upper()} Evidence: {result}")
        
        evidence_text = "\n\n".join(evidence_parts)
        
        synthesis_prompt = f"""Query: {query}

Evidence from multiple sources:
{evidence_text}

Synthesize the evidence and provide a comprehensive answer to the query."""
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        return {
            'query': query,
            'modality_results': modality_results,
            'synthesized_answer': response.content[0].text
        }
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents"""
        return {
            name: agent.capabilities
            for name, agent in self.agents.items()
        }