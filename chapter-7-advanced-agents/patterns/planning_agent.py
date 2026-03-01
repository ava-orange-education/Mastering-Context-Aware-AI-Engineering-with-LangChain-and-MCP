"""
Planning agent that decomposes tasks into steps and executes them.
"""

from typing import Dict, Any, List, Optional
from ..core.agent_base import Agent, AgentAction, AgentObservation
from ..core.agent_state import AgentState
import logging
import json

logger = logging.getLogger(__name__)


class PlanningAgent(Agent):
    """
    Agent that creates a plan before execution.
    
    Pattern: Plan -> Execute Step 1 -> Execute Step 2 -> ... -> Complete
    """
    
    def __init__(self, name: str, llm_client, tools: Optional[List] = None):
        super().__init__(name, llm_client, tools)
        self.plan: List[Dict[str, Any]] = []
        self.current_plan_step = 0
    
    def create_plan(self, task: str) -> List[Dict[str, Any]]:
        """Generate execution plan for task"""
        tools_desc = self.get_available_tools_description()
        
        prompt = f"""Create a detailed step-by-step plan to accomplish this task:

Task: {task}

Available tools:
{tools_desc}

Generate a plan as a JSON list of steps. Each step should have:
- step_number: integer
- description: what to do
- tool: which tool to use (or null if no tool needed)
- expected_output: what this step should produce

Return ONLY the JSON array, no other text.

Example format:
[
  {{"step_number": 1, "description": "Search for information", "tool": "search", "expected_output": "relevant documents"}},
  {{"step_number": 2, "description": "Analyze results", "tool": null, "expected_output": "summary of findings"}}
]"""
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=messages
        )
        
        # Parse JSON from response
        response_text = response.content[0].text
        
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        try:
            plan = json.loads(response_text)
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            # Fallback to simple plan
            return [
                {
                    "step_number": 1,
                    "description": task,
                    "tool": None,
                    "expected_output": "task completion"
                }
            ]
    
    def think(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Think about next step in plan"""
        if not self.plan:
            return f"I need to create a plan for: {input_text}"
        
        if self.current_plan_step >= len(self.plan):
            return "All plan steps completed. Ready to finish."
        
        current_step = self.plan[self.current_plan_step]
        
        # Build context from previous steps
        previous_results = []
        for i, step in enumerate(self.execution_history):
            if step.observation:
                previous_results.append(f"Step {i+1}: {step.observation.result}")
        
        context_text = "\n".join(previous_results) if previous_results else "No previous results"
        
        thought = f"""Executing plan step {current_step['step_number']}: {current_step['description']}

Previous results:
{context_text}

Expected output: {current_step['expected_output']}"""
        
        return thought
    
    def act(self, thought: str) -> AgentAction:
        """Determine action based on current plan step"""
        if not self.plan:
            return AgentAction(
                action_type="plan",
                reasoning="Need to create execution plan"
            )
        
        if self.current_plan_step >= len(self.plan):
            return AgentAction(
                action_type="finish",
                tool_input={"final_answer": "Plan completed successfully"}
            )
        
        current_step = self.plan[self.current_plan_step]
        
        if current_step['tool']:
            return AgentAction(
                action_type=current_step['tool'],
                tool_name=current_step['tool'],
                tool_input={"query": current_step['description']},
                reasoning=f"Executing step {current_step['step_number']}"
            )
        else:
            # No tool needed, use LLM for this step
            return AgentAction(
                action_type="think",
                reasoning=current_step['description']
            )
    
    def observe(self, action: AgentAction) -> AgentObservation:
        """Execute action and observe result"""
        if action.action_type == "plan":
            # Create the plan
            self.plan = self.create_plan(action.reasoning or "")
            return AgentObservation(
                result=f"Created plan with {len(self.plan)} steps",
                success=True,
                metadata={'plan': self.plan}
            )
        
        if action.action_type == "finish":
            return AgentObservation(
                result=action.tool_input.get('final_answer', 'Completed'),
                success=True
            )
        
        if action.action_type == "think":
            # Use LLM to process this step
            prompt = f"Task: {action.reasoning}\n\nProvide a detailed response."
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages
            )
            
            result = response.content[0].text
            self.current_plan_step += 1
            
            return AgentObservation(
                result=result,
                success=True
            )
        
        # Execute tool
        tool = self._find_tool(action.tool_name)
        
        if not tool:
            self.current_plan_step += 1
            return AgentObservation(
                result=f"Tool {action.tool_name} not found, skipping step",
                success=False,
                error=f"Unknown tool: {action.tool_name}"
            )
        
        try:
            result = tool.execute(action.tool_input)
            self.current_plan_step += 1
            
            return AgentObservation(
                result=result,
                success=True,
                metadata={'tool': tool.name, 'step': self.current_plan_step}
            )
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            self.current_plan_step += 1
            
            return AgentObservation(
                result=f"Error in step {self.current_plan_step}: {str(e)}",
                success=False,
                error=str(e)
            )
    
    def _find_tool(self, tool_name: str):
        """Find tool by name"""
        if not tool_name:
            return None
        
        for tool in self.tools:
            if tool.name.lower() == tool_name.lower():
                return tool
        return None
    
    def run_with_planning(self, task: str, max_steps: int = 20) -> Dict[str, Any]:
        """Execute task with explicit planning"""
        logger.info(f"PlanningAgent {self.name} starting task: {task}")
        
        # Create plan
        self.plan = self.create_plan(task)
        logger.info(f"Created plan with {len(self.plan)} steps")
        
        # Execute plan
        result = self.run(task, max_steps=max_steps)
        
        # Add plan to result
        result['plan'] = self.plan
        result['steps_completed'] = self.current_plan_step
        
        return result 