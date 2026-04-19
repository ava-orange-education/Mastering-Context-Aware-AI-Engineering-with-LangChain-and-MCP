"""
Answer Relevance Evaluator

Evaluates relevance and quality of generated answers
"""

from typing import List, Dict, Any, Optional
import logging
import sys
sys.path.append('../..')

from shared.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AnswerRelevanceEvaluator:
    """
    Evaluator for answer relevance and quality
    """
    
    def __init__(self):
        self.llm = BaseAgent(
            name="Answer Evaluator",
            model="claude-sonnet-4-20250514",
            temperature=0.0
        )
    
    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate answer quality
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_docs: Retrieved documents used
            ground_truth: Optional reference answer
        
        Returns:
            Evaluation metrics
        """
        
        # Evaluate different aspects
        relevance_score = await self._evaluate_relevance(query, answer)
        faithfulness_score = await self._evaluate_faithfulness(answer, retrieved_docs)
        completeness_score = await self._evaluate_completeness(query, answer)
        
        # Compare to ground truth if available
        correctness_score = None
        if ground_truth:
            correctness_score = await self._evaluate_correctness(answer, ground_truth)
        
        results = {
            "relevance_score": relevance_score,
            "faithfulness_score": faithfulness_score,
            "completeness_score": completeness_score,
            "correctness_score": correctness_score,
            "overall_score": self._calculate_overall_score(
                relevance_score,
                faithfulness_score,
                completeness_score,
                correctness_score
            )
        }
        
        logger.info(f"Answer evaluation: {results['overall_score']:.2f}")
        
        return results
    
    async def _evaluate_relevance(
        self,
        query: str,
        answer: str
    ) -> float:
        """Evaluate if answer is relevant to query"""
        
        prompt = f"""Evaluate how relevant this answer is to the query on a scale of 0.0 to 1.0.

Query: {query}

Answer: {answer}

Respond with just a number between 0.0 and 1.0, where:
- 1.0 = Perfectly relevant, directly answers the query
- 0.5 = Somewhat relevant but incomplete or tangential
- 0.0 = Not relevant at all

Score:"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm._call_llm(messages)
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default if parsing fails
    
    async def _evaluate_faithfulness(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> float:
        """Evaluate if answer is faithful to source documents"""
        
        # Format documents
        docs_text = "\n\n".join([
            f"Document {i+1}: {doc.get('content', '')[:200]}..."
            for i, doc in enumerate(retrieved_docs[:3])
        ])
        
        prompt = f"""Evaluate if this answer is faithful to the source documents on a scale of 0.0 to 1.0.

Source Documents:
{docs_text}

Answer: {answer}

Respond with just a number between 0.0 and 1.0, where:
- 1.0 = All statements are supported by the documents
- 0.5 = Some statements supported, some not verifiable
- 0.0 = Contains information not in documents or contradicts them

Score:"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm._call_llm(messages)
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    async def _evaluate_completeness(
        self,
        query: str,
        answer: str
    ) -> float:
        """Evaluate if answer completely addresses the query"""
        
        prompt = f"""Evaluate how completely this answer addresses the query on a scale of 0.0 to 1.0.

Query: {query}

Answer: {answer}

Respond with just a number between 0.0 and 1.0, where:
- 1.0 = Fully addresses all aspects of the query
- 0.5 = Partially addresses the query, missing some aspects
- 0.0 = Does not address the query at all

Score:"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm._call_llm(messages)
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    async def _evaluate_correctness(
        self,
        answer: str,
        ground_truth: str
    ) -> float:
        """Evaluate correctness against ground truth"""
        
        prompt = f"""Evaluate how correct this answer is compared to the reference answer on a scale of 0.0 to 1.0.

Reference Answer: {ground_truth}

Generated Answer: {answer}

Respond with just a number between 0.0 and 1.0, where:
- 1.0 = Semantically equivalent to reference
- 0.5 = Partially correct
- 0.0 = Incorrect or contradicts reference

Score:"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm._call_llm(messages)
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_overall_score(
        self,
        relevance: float,
        faithfulness: float,
        completeness: float,
        correctness: Optional[float]
    ) -> float:
        """Calculate overall score"""
        
        scores = [relevance, faithfulness, completeness]
        weights = [0.35, 0.35, 0.3]
        
        if correctness is not None:
            scores.append(correctness)
            weights = [0.25, 0.25, 0.25, 0.25]
        
        overall = sum(s * w for s, w in zip(scores, weights))
        
        return round(overall, 3)