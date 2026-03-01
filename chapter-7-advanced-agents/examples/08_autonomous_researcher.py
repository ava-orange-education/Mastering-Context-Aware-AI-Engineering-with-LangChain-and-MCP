"""
Example 8: Autonomous research agent that gathers and synthesizes information.
"""

import sys
sys.path.append('..')

from patterns.planning_agent import PlanningAgent
from patterns.reflection_agent import ReflectionAgent
from tools.search_tools import WebSearchTool, VectorSearchTool
from tools.file_tools import WriteFileTool
from memory.memory_manager import UnifiedMemoryManager
from embedding.embedding_manager import EmbeddingManager
from retrieval.vector_store import ChromaVectorStore
from anthropic import Anthropic
import os


class ResearchAgent(PlanningAgent):
    """Agent specialized for research tasks"""
    
    def __init__(self, name: str, llm_client, tools, memory_manager):
        super().__init__(name, llm_client, tools)
        self.memory = memory_manager
        self.research_findings = []
    
    def conduct_research(self, research_question: str, depth: int = 3) -> Dict[str, Any]:
        """
        Conduct research on a question
        
        Args:
            research_question: Question to research
            depth: Research depth (number of iterations)
            
        Returns:
            Research results
        """
        print(f"Starting research on: {research_question}\n")
        
        findings = []
        
        for iteration in range(depth):
            print(f"Research iteration {iteration + 1}/{depth}")
            
            # Create research plan for this iteration
            plan = self.create_plan(
                f"Find information about: {research_question}. "
                f"Previous findings: {self._summarize_findings(findings)}"
            )
            
            # Execute research
            result = self.run_with_planning(research_question, max_steps=10)
            
            # Extract findings
            if result['success']:
                findings.append({
                    'iteration': iteration + 1,
                    'result': result['result'],
                    'steps': result['total_steps']
                })
                
                # Store in memory
                self.memory.episodic.store_experience({
                    'content': f"Research on: {research_question}",
                    'outcome': result['result'],
                    'success': True,
                    'type': 'research',
                    'metadata': {'iteration': iteration + 1}
                })
        
        # Synthesize findings
        synthesis = self._synthesize_findings(research_question, findings)
        
        return {
            'research_question': research_question,
            'findings': findings,
            'synthesis': synthesis,
            'depth': depth
        }
    
    def _summarize_findings(self, findings: List[Dict]) -> str:
        """Summarize current findings"""
        if not findings:
            return "No previous findings"
        
        return ". ".join([f"Iteration {f['iteration']}: {str(f['result'])[:100]}" 
                         for f in findings])
    
    def _synthesize_findings(self, question: str, findings: List[Dict]) -> str:
        """Synthesize research findings"""
        findings_text = "\n\n".join([
            f"Finding {f['iteration']}:\n{f['result']}"
            for f in findings
        ])
        
        synthesis_prompt = f"""Synthesize these research findings into a comprehensive answer:

Research Question: {question}

Findings:
{findings_text}

Provide a well-structured synthesis that:
1. Directly answers the research question
2. Integrates insights from all findings
3. Notes any conflicting information
4. Identifies gaps in current knowledge"""
        
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        return response.content[0].text


def main():
    print("=== Autonomous Research Agent ===\n")
    
    # Initialize components
    llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    embedding_mgr = EmbeddingManager(
        model_name="text-embedding-3-small",
        provider="openai"
    )
    
    vector_store = ChromaVectorStore(
        collection_name="research_agent",
        dimensions=1536
    )
    
    memory = UnifiedMemoryManager(
        agent_id="researcher_001",
        llm_client=llm_client,
        embedding_manager=embedding_mgr,
        vector_store=vector_store
    )
    
    # Create tools
    search_tool = WebSearchTool()
    
    vector_search = VectorSearchTool(
        vector_store=vector_store,
        embedding_manager=embedding_mgr
    )
    
    write_tool = WriteFileTool(allowed_paths=["./output"])
    
    # Create research agent
    agent = ResearchAgent(
        name="ResearchAgent",
        llm_client=llm_client,
        tools=[search_tool, vector_search, write_tool],
        memory_manager=memory
    )
    
    # Conduct research
    research_question = "What are the latest developments in large language model reasoning capabilities?"
    
    results = agent.conduct_research(research_question, depth=2)
    
    print("\n" + "=" * 60)
    print("\nResearch Complete!\n")
    print(f"Question: {results['research_question']}")
    print(f"\nSynthesis:\n{results['synthesis']}")
    
    # Save results
    output_file = "./output/research_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"Research Question: {results['research_question']}\n\n")
        f.write(f"Synthesis:\n{results['synthesis']}\n\n")
        f.write(f"Detailed Findings:\n")
        for finding in results['findings']:
            f.write(f"\nIteration {finding['iteration']}:\n{finding['result']}\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()