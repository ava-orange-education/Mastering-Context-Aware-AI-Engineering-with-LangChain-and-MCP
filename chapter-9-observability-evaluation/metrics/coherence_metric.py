"""
Coherence metric to measure logical flow and consistency of responses.
"""

from typing import Dict, Any, List
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoherenceMetric:
    """Measure response coherence and logical flow"""
    
    def __init__(self, llm_client=None):
        """
        Initialize coherence metric
        
        Args:
            llm_client: Optional LLM client for evaluation
        """
        self.llm = llm_client
    
    def calculate(self, response: str) -> Dict[str, Any]:
        """
        Calculate coherence score
        
        Args:
            response: Generated response
            
        Returns:
            Coherence metrics
        """
        # Check various coherence aspects
        structure_score = self._check_structure(response)
        repetition_score = self._check_repetition(response)
        transition_score = self._check_transitions(response)
        
        # Overall coherence score (average of components)
        coherence_score = (structure_score + repetition_score + transition_score) / 3
        
        return {
            'coherence_score': coherence_score,
            'structure_score': structure_score,
            'repetition_score': repetition_score,
            'transition_score': transition_score,
            'method': 'rule_based'
        }
    
    def _check_structure(self, text: str) -> float:
        """
        Check structural coherence
        
        Args:
            text: Text to analyze
            
        Returns:
            Structure score (0-1)
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0.0
        
        # Check for reasonable sentence length variation
        lengths = [len(s.split()) for s in sentences]
        
        if not lengths:
            return 0.0
        
        avg_length = sum(lengths) / len(lengths)
        
        # Penalize if all sentences are very short or very long
        if avg_length < 3:
            return 0.5  # Too terse
        elif avg_length > 50:
            return 0.7  # Too verbose
        
        # Good structure
        return 1.0
    
    def _check_repetition(self, text: str) -> float:
        """
        Check for excessive repetition
        
        Args:
            text: Text to analyze
            
        Returns:
            Repetition score (0-1, higher is better)
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0
        
        # Check for exact sentence repetition
        unique_sentences = len(set(sentences))
        repetition_ratio = unique_sentences / len(sentences)
        
        # Check for phrase repetition
        phrases = []
        for sentence in sentences:
            words = sentence.split()
            # Extract 3-word phrases
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrases.append(phrase)
        
        if phrases:
            unique_phrases = len(set(phrases))
            phrase_repetition_ratio = unique_phrases / len(phrases)
        else:
            phrase_repetition_ratio = 1.0
        
        # Combine sentence and phrase repetition
        score = (repetition_ratio + phrase_repetition_ratio) / 2
        
        return max(0.0, min(1.0, score))
    
    def _check_transitions(self, text: str) -> float:
        """
        Check for logical transitions between sentences
        
        Args:
            text: Text to analyze
            
        Returns:
            Transition score (0-1)
        """
        # Transition words/phrases
        transitions = [
            'however', 'moreover', 'furthermore', 'additionally',
            'therefore', 'thus', 'consequently', 'as a result',
            'for example', 'for instance', 'such as', 'specifically',
            'in contrast', 'on the other hand', 'similarly',
            'first', 'second', 'third', 'finally', 'in conclusion'
        ]
        
        text_lower = text.lower()
        
        # Count transition usage
        transition_count = sum(1 for t in transitions if t in text_lower)
        
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count <= 1:
            return 1.0
        
        # Expect roughly one transition per 3-4 sentences
        expected_transitions = sentence_count / 3.5
        
        if transition_count >= expected_transitions:
            return 1.0
        elif transition_count >= expected_transitions * 0.5:
            return 0.8
        else:
            return 0.6
    
    def calculate_with_llm(self, response: str) -> Dict[str, Any]:
        """
        Calculate coherence using LLM evaluation
        
        Args:
            response: Generated response
            
        Returns:
            Coherence metrics
        """
        if not self.llm:
            return self.calculate(response)
        
        prompt = f"""Evaluate the coherence of this response on a scale of 0-1.

Response:
{response}

Assess:
1. Logical flow between ideas
2. Consistency of arguments
3. Clear structure and organization
4. Appropriate use of transitions
5. Lack of contradictions

Provide score in format: COHERENCE_SCORE: [0.0-1.0]
Brief explanation: [your reasoning]"""
        
        try:
            result = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = result.content[0].text
            
            score_match = re.search(r'COHERENCE_SCORE:\s*(0?\.\d+|1\.0)', result_text)
            score = float(score_match.group(1)) if score_match else 0.5
            
            return {
                'coherence_score': score,
                'method': 'llm_evaluation',
                'explanation': result_text
            }
        
        except Exception as e:
            logger.error(f"LLM coherence check failed: {e}")
            return self.calculate(response) 