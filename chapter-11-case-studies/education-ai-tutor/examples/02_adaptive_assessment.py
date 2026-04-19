"""
Example 2: Adaptive Assessment

Demonstrates adaptive assessment and feedback generation
"""

import asyncio
import sys
sys.path.append('../..')

from agents.assessment_agent import AssessmentAgent
from agents.feedback_agent import FeedbackAgent
from agents.adaptation_agent import AdaptationAgent
from student_model.knowledge_state import KnowledgeState
from personalization.difficulty_scaler import DifficultyScaler
from shared.utils import setup_logging

setup_logging()


async def basic_assessment():
    """Basic assessment example"""
    
    print("\n" + "="*70)
    print("Example 1: Basic Assessment")
    print("="*70)
    
    assessor = AssessmentAgent()
    
    # Sample assessment
    questions = [
        {
            "question": "Solve for x: 2x + 5 = 13",
            "correct_answer": "x = 4"
        },
        {
            "question": "Simplify: 3(x + 2) - 2x",
            "correct_answer": "x + 6"
        },
        {
            "question": "What is the slope of the line y = 3x + 2?",
            "correct_answer": "3"
        }
    ]
    
    student_answers = [
        {"answer": "x = 4"},
        {"answer": "x + 6"},
        {"answer": "3"}
    ]
    
    print("\nAssessing student on algebra...")
    
    result = await assessor.process({
        "student_id": "student_001",
        "topic": "algebra",
        "questions": questions,
        "student_answers": student_answers
    })
    
    print("\n" + "-"*70)
    print("Assessment Results")
    print("-"*70)
    print(result.content)
    
    metrics = result.metadata.get("metrics", {})
    print("\n" + "-"*70)
    print("Metrics")
    print("-"*70)
    print(f"  Overall Score: {metrics.get('average_score', 0):.1%}")
    print(f"  Mastery Level: {metrics.get('mastery_level', 0):.1%}")
    print(f"  Correct: {metrics.get('correct_count', 0)}/{metrics.get('total_questions', 0)}")


async def detailed_feedback_example():
    """Detailed feedback on student work"""
    
    print("\n" + "="*70)
    print("Example 2: Detailed Feedback")
    print("="*70)
    
    feedback_agent = FeedbackAgent()
    
    # Student's work
    student_work = {
        "question": "Explain the water cycle",
        "answer": """Water evaporates from oceans and lakes. It goes up into the sky and forms clouds. 
        Then it rains and the water goes back to the ground.""",
        "work_shown": "Drew a simple diagram with arrows"
    }
    
    print("\nStudent's Answer:")
    print(f"  {student_work['answer']}")
    
    result = await feedback_agent.process({
        "student_work": student_work,
        "learning_goals": ["Understand water cycle stages", "Use scientific vocabulary"]
    })
    
    print("\n" + "-"*70)
    print("Detailed Feedback")
    print("-"*70)
    print(result.content)


async def adaptive_difficulty_assessment():
    """Assessment that adapts difficulty based on performance"""
    
    print("\n" + "="*70)
    print("Example 3: Adaptive Difficulty Assessment")
    print("="*70)
    
    adapter = AdaptationAgent()
    scaler = DifficultyScaler()
    
    # Simulate performance history
    performance_history = [
        {"accuracy": 0.9, "time_taken": 45, "correct": True},
        {"accuracy": 0.95, "time_taken": 40, "correct": True},
        {"accuracy": 0.9, "time_taken": 42, "correct": True},
        {"accuracy": 1.0, "time_taken": 35, "correct": True},
    ]
    
    print("\nStudent Performance History:")
    for i, perf in enumerate(performance_history, 1):
        print(f"  Problem {i}: {perf['accuracy']:.0%} in {perf['time_taken']}s")
    
    # Get adaptation recommendation
    result = await adapter.process({
        "student_id": "student_high_performer",
        "topic": "algebra",
        "performance_history": performance_history,
        "current_difficulty": "medium",
        "time_spent": 180,
        "errors": []
    })
    
    print("\n" + "-"*70)
    print("Adaptation Recommendation")
    print("-"*70)
    print(result.content)
    
    # Get difficulty recommendation
    difficulty_rec = scaler.recommend_difficulty(
        current_difficulty="medium",
        recent_performance=performance_history
    )
    
    print("\n" + "-"*70)
    print("Difficulty Adjustment")
    print("-"*70)
    print(f"  Current: {difficulty_rec['current_difficulty']}")
    print(f"  Recommended: {difficulty_rec['recommended_difficulty']}")
    print(f"  Change: {difficulty_rec['change']}")
    print(f"  Reason: {difficulty_rec['reason']}")


async def struggling_student_assessment():
    """Assessment for struggling student with support"""
    
    print("\n" + "="*70)
    print("Example 4: Struggling Student Assessment")
    print("="*70)
    
    assessor = AssessmentAgent()
    feedback_agent = FeedbackAgent()
    
    # Student struggling with fractions
    questions = [
        {
            "question": "What is 1/2 + 1/4?",
            "correct_answer": "3/4"
        },
        {
            "question": "What is 2/3 × 3/4?",
            "correct_answer": "1/2"
        },
        {
            "question": "Simplify: 6/8",
            "correct_answer": "3/4"
        }
    ]
    
    student_answers = [
        {"answer": "2/6"},  # Incorrect
        {"answer": "5/7"},  # Incorrect
        {"answer": "3/4"}   # Correct
    ]
    
    print("\nAssessing struggling student on fractions...")
    
    result = await assessor.process({
        "student_id": "struggling_student",
        "topic": "fractions",
        "questions": questions,
        "student_answers": student_answers
    })
    
    print("\n" + "-"*70)
    print("Assessment Results")
    print("-"*70)
    
    metrics = result.metadata.get("metrics", {})
    print(f"  Score: {metrics.get('average_score', 0):.1%}")
    print(f"  Mastery: {metrics.get('mastery_level', 0):.1%}")
    
    # Generate encouraging feedback
    print("\n" + "-"*70)
    print("Supportive Feedback")
    print("-"*70)
    
    feedback_result = await feedback_agent.process({
        "student_work": {
            "topic": "fractions",
            "performance": "struggling",
            "score": metrics.get('average_score', 0)
        },
        "student_level": "6th_grade"
    })
    
    print(feedback_result.content)


async def mastery_based_progression():
    """Assessment determines if student ready to progress"""
    
    print("\n" + "="*70)
    print("Example 5: Mastery-Based Progression")
    print("="*70)
    
    knowledge_state = KnowledgeState("student_progression")
    
    # Topic progression
    topics = [
        ("addition", 0.9),
        ("subtraction", 0.85),
        ("multiplication", 0.75),
        ("division", 0.6)
    ]
    
    print("\nStudent Mastery Levels:")
    for topic, mastery in topics:
        knowledge_state.mastery_levels[topic] = mastery
        status = "✓ Mastered" if mastery >= 0.8 else "◐ In Progress" if mastery >= 0.6 else "○ Not Ready"
        print(f"  {topic.title()}: {mastery:.0%} {status}")
    
    # Check readiness for advanced topic
    knowledge_state.set_prerequisites("fractions", ["multiplication", "division"])
    
    ready = knowledge_state.is_ready_for("fractions")
    
    print("\n" + "-"*70)
    print("Progression Check: Fractions")
    print("-"*70)
    print(f"  Prerequisites: multiplication, division")
    print(f"  Ready: {'Yes' if ready else 'No'}")
    
    if not ready:
        print("\n  Recommendation: Master division before starting fractions")
        print("  Suggested practice: Division problems at current level")


async def formative_assessment():
    """Ongoing formative assessment during learning"""
    
    print("\n" + "="*70)
    print("Example 6: Formative Assessment")
    print("="*70)
    
    feedback_agent = FeedbackAgent()
    
    # Student working through a problem
    problem_attempts = [
        {
            "step": "Identify the equation",
            "student_work": "2x + 3 = 11",
            "correct": True
        },
        {
            "step": "Isolate the variable term",
            "student_work": "2x = 11 + 3",
            "correct": False
        }
    ]
    
    print("\nStudent Working Through Problem:")
    for attempt in problem_attempts:
        status = "✓" if attempt["correct"] else "✗"
        print(f"  {status} {attempt['step']}: {attempt['student_work']}")
    
    # Get immediate feedback
    result = await feedback_agent.process({
        "student_work": {
            "problem": "Solve 2x + 3 = 11",
            "attempt": problem_attempts[1]["student_work"],
            "step": problem_attempts[1]["step"]
        },
        "student_level": "8th_grade"
    })
    
    print("\n" + "-"*70)
    print("Immediate Formative Feedback")
    print("-"*70)
    print(result.content)


async def hint_generation():
    """Generate hints for struggling student"""
    
    print("\n" + "="*70)
    print("Example 7: Adaptive Hint Generation")
    print("="*70)
    
    feedback_agent = FeedbackAgent()
    
    problem = "Find the area of a triangle with base 8 cm and height 5 cm"
    student_attempt = "I multiplied 8 × 5 = 40, so the answer is 40 cm²"
    
    print(f"\nProblem: {problem}")
    print(f"Student's Attempt: {student_attempt}")
    
    # Generate hints at different levels
    hint_levels = ["light", "medium", "strong"]
    
    for level in hint_levels:
        print(f"\n{'-'*70}")
        print(f"{level.upper()} Hint")
        print(f"{'-'*70}")
        
        hint = await feedback_agent.generate_hint(
            problem=problem,
            student_attempt=student_attempt,
            hint_level=level
        )
        
        print(hint)


async def main():
    """Run all adaptive assessment examples"""
    
    print("\n" + "="*70)
    print("Adaptive Assessment Examples")
    print("="*70)
    
    await basic_assessment()
    
    await detailed_feedback_example()
    
    await adaptive_difficulty_assessment()
    
    await struggling_student_assessment()
    
    await mastery_based_progression()
    
    await formative_assessment()
    
    await hint_generation()
    
    print("\n" + "="*70)
    print("All Adaptive Assessment Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())