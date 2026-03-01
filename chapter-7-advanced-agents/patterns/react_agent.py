"""
ReAct (Reasoning + Acting) agent pattern implementation.
"""

from typing import Dict, Any, List, Optional
import re
from ..core.agent_base import Agent, AgentAction, AgentObservation
import logging

logger = logging.getLogger(__name__)


class ReActAgent(Agent):
    """
    ReAct agent that interleaves reasoning (thinking) and acting (tool use).
    
    ReAct pattern: Thought -> Action -> Observation -> Thought -> ...
    """
    
    def __init__(self, name: str, llm_client, tools: Optional[List] = None):
        super().__init__(name, llm_client, tools)
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build ReAct system prompt"""
        tools_desc = self.get_available_tools_description()
        
        return f"""You are a helpful AI assistant that solves tasks using the ReAct pattern.

You have access to the following tools:
{tools_desc}

Use this format:

Thought: Consider what to do next
Action: choose one action from [{', '.join([t.name for t in self.tools])}, finish]
Action Input: the input for the action
Observation: the result of the action

... (repeat Thought/Action/Observation as needed)

Thought: I now know the final answer
Action: finish
Action Input: {{"final_answer": "your final answer here"}}

Important:
- Always start with a Thought
- Each Action must be followed by an Observation before the next Thought
- Use "finish" action when you have the final answer
- Be concise and focused in your thoughts"""
    
    def think(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate thought using LLM"""
        # Build conversation history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add execution history
        for step in self.execution_history:
            messages.append({
                "role": "assistant",
                "content": f"Thought: {step.thought}\nAction: {step.action.action_type}\nAction Input: {step.action.tool_input}"
            })
            if step.observation:
                messages.append({
                    "role": "user",
                    "content": f"Observation: {step.observation.result}"
                })
        
        # Add current input
        messages.append({"role": "user", "content": input_text})
        
        # Generate response
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=messages
        )
        
        return response.content[0].text
    
    def act(self, thought: str) -> AgentAction:
        """Parse action from thought"""
        # Extract action using regex
        action_pattern = r"Action:\s*(.+?)(?:\n|$)"
        action_input_pattern = r"Action Input:\s*(.+?)(?:\n|$)"
        
        action_match = re.search(action_pattern, thought, re.IGNORECASE)
        action_input_match = re.search(action_input_pattern, thought, re.IGNORECASE | re.DOTALL)
        
        if not action_match:
            # If no explicit action, extract thought and continue
            thought_pattern = r"Thought:\s*(.+?)(?:\n|$)"
            thought_match = re.search(thought_pattern, thought, re.IGNORECASE)
            thought_text = thought_match.group(1).strip() if thought_match else thought
            
            return AgentAction(
                action_type="continue",
                reasoning=thought_text
            )
        
        action_type = action_match.group(1).strip()
        
        # Parse action input
        action_input = {}
        if action_input_match:
            input_text = action_input_match.group(1).strip()
            try:
                import json
                action_input = json.loads(input_text)
            except:
                action_input = {"input": input_text}
        
        # Extract reasoning (the Thought part)
        thought_pattern = r"Thought:\s*(.+?)(?:\n|$)"
        thought_match = re.search(thought_pattern, thought, re.IGNORECASE)
        reasoning = thought_match.group(1).strip() if thought_match else None
        
        return AgentAction(
            action_type=action_type.lower(),
            tool_name=action_type if action_type.lower() != "finish" else None,
            tool_input=action_input,
            reasoning=reasoning
        )
    
    def observe(self, action: AgentAction) -> AgentObservation:
        """Execute action and observe result"""
        # Handle finish action
        if action.action_type == "finish":
            return AgentObservation(
                result=action.tool_input.get('final_answer', 'Task completed'),
                success=True
            )
        
        # Find and execute tool
        tool = self._find_tool(action.action_type)
        
        if not tool:
            return AgentObservation(
                result=f"Error: Tool '{action.action_type}' not found",
                success=False,
                error=f"Unknown tool: {action.action_type}"
            )
        
        try:
            result = tool.execute(action.tool_input)
            return AgentObservation(
                result=result,
                success=True,
                metadata={'tool': tool.name}
            )
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return AgentObservation(
                result=f"Error executing {tool.name}: {str(e)}",
                success=False,
                error=str(e)
            )
    
    def _find_tool(self, tool_name: str):
        """Find tool by name"""
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                return tool
        return None