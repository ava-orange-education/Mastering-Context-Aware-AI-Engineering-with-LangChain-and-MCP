"""
Reflection agent that self-evaluates and improves its responses.
"""

from typing import Dict, Any, List, Optional
from ..core.agent_base import Agent, AgentAction, AgentObservation
import logging

logger = logging.getLogger(__name__)


class ReflectionAgent(Agent):
    """
    Agent that reflects on its outputs and iteratively improves them.
    
    Pattern: Generate -> Reflect -> Refine -> Generate -> ...
    """
    
    def __init__(self, name: str, llm_client, tools: Optional[List] = None,
                 max_reflections: int = 3):
        super().__init__(name, llm_client, tools)
        self.max_reflections = max_reflections
        self.reflection_history: List[Dict[str, str]] = []
    
    def think(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate initial response or refined response"""
        if not self.reflection_history:
            # Initial generation
            prompt = f"""Task: {input_text}

Generate a comprehensive response to this task. Be thorough and accurate."""
            
            messages = [{"role": "user", "content": prompt}]
        else:
            # Refinement based on reflection
            last_reflection = self.reflection_history[-1]
            
            prompt = f"""Original task: {input_text}

Previous response:
{last_reflection['response']}

Reflection on previous response:
{last_reflection['reflection']}

Based on the reflection, generate an improved response that addresses the identified issues."""
            
            messages = [{"role": "user", "content": prompt}]
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=messages
        )
        
        return response.content[0].text
    
    def reflect(self, response: str) -> str:
        """Reflect on the quality of the response"""
        prompt = f"""Critically evaluate this response:

{response}

Provide a detailed reflection covering:
1. Accuracy: Is the information correct?
2. Completeness: Are all aspects addressed?
3. Clarity: Is it easy to understand?
4. Conciseness: Is it appropriately detailed without being verbose?
5. Specific improvements: What exactly should be changed?

Be specific about issues and suggest concrete improvements."""
        
        messages = [{"role": "user", "content": prompt}]
        
        reflection_response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages
        )
        
        return reflection_response.content[0].text
    
    def act(self, thought: str) -> AgentAction:
        """Determine if refinement is needed"""
        return AgentAction(
            action_type="reflect",
            tool_input={"response": thought}
        )
    
    def observe(self, action: AgentAction) -> AgentObservation:
        """Execute reflection"""
        response = action.tool_input.get('response', '')
        
        if len(self.reflection_history) >= self.max_reflections:
            return AgentObservation(
                result="Max reflections reached. Finalizing response.",
                success=True,
                metadata={'final': True}
            )
        
        # Generate reflection
        reflection = self.reflect(response)
        
        # Store in history
        self.reflection_history.append({
            'response': response,
            'reflection': reflection,
            'iteration': len(self.reflection_history) + 1
        })
        
        return AgentObservation(
            result=reflection,
            success=True,
            metadata={'iteration': len(self.reflection_history)}
        )
    
    def run_with_reflection(self, task: str) -> Dict[str, Any]:
        """Execute task with reflection loop"""
        logger.info(f"ReflectionAgent {self.name} starting task: {task}")
        
        best_response = None
        
        for iteration in range(self.max_reflections + 1):
            # Generate response
            response = self.think(task)
            logger.info(f"Iteration {iteration + 1} - Generated response")
            
            if iteration == self.max_reflections:
                # Final iteration, no more reflections
                best_response = response
                break
            
            # Reflect on response
            reflection = self.reflect(response)
            logger.info(f"Iteration {iteration + 1} - Reflection: {reflection[:200]}...")
            
            # Store reflection
            self.reflection_history.append({
                'response': response,
                'reflection': reflection,
                'iteration': iteration + 1
            })
            
            # Check if good enough (simplified check)
            if "no significant issues" in reflection.lower() or "excellent" in reflection.lower():
                best_response = response
                logger.info(f"Response quality sufficient at iteration {iteration + 1}")
                break
        
        return {
            'success': True,
            'final_response': best_response,
            'iterations': len(self.reflection_history),
            'reflection_history': self.reflection_history
        }


class SelfCritiqueAgent(ReflectionAgent):
    """Agent that critiques its own reasoning process"""
    
    def critique_reasoning(self, reasoning_chain: List[str]) -> Dict[str, Any]:
        """Critique a chain of reasoning steps"""
        reasoning_text = "\n".join([f"Step {i+1}: {step}" 
                                   for i, step in enumerate(reasoning_chain)])
        
        prompt = f"""Analyze this reasoning chain for logical errors or weaknesses:

{reasoning_text}

Identify:
1. Logical fallacies or errors
2. Unsupported assumptions
3. Missing steps in reasoning
4. Alternative approaches that might work better

Provide specific, actionable critique."""
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages
        )
        
        critique = response.content[0].text
        
        return {
            'critique': critique,
            'reasoning_chain': reasoning_chain,
            'has_issues': any(word in critique.lower() 
                            for word in ['error', 'fallacy', 'assumption', 'missing'])
        }