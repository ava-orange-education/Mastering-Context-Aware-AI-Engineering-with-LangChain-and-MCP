"""
Example 1: Personalized Lesson

Demonstrates personalized teaching adapted to student's learning style and level
"""

import asyncio
import sys
sys.path.append('../..')

from agents.teaching_agent import TeachingAgent
from agents.explanation_agent import ExplanationAgent
from student_model.knowledge_state import KnowledgeState
from student_model.learning_style import LearningStyleAnalyzer
from shared.utils import setup_logging

setup_logging()


async def basic_personalized_lesson():
    """Basic personalized lesson example"""
    
    print("\n" + "="*70)
    print("Example 1: Basic Personalized Lesson")
    print("="*70)
    
    # Initialize teaching agent
    teacher = TeachingAgent()
    
    # Create student profile
    student_profile = {
        "student_id": "student_001",
        "grade_level": "9th grade",
        "learning_style": "visual",
        "interests": ["space", "technology"],
        "mastery_levels": {
            "basic_algebra": 0.8,
            "linear_equations": 0.6
        }
    }
    
    print("\nStudent Profile:")
    print(f"  Grade Level: {student_profile['grade_level']}")
    print(f"  Learning Style: {student_profile['learning_style']}")
    print(f"  Interests: {', '.join(student_profile['interests'])}")
    
    # Request lesson
    print("\nRequesting lesson on quadratic equations...")
    
    result = await teacher.process({
        "student_profile": student_profile,
        "topic": "quadratic equations",
        "question": "What are quadratic equations and how do I solve them?"
    })
    
    print("\n" + "-"*70)
    print("Personalized Lesson")
    print("-"*70)
    print(result.content)
    
    print("\n" + "-"*70)
    print("Teaching Elements Used")
    print("-"*70)
    elements = result.metadata.get("teaching_elements", {})
    for element, present in elements.items():
        status = "✓" if present else "✗"
        print(f"  {status} {element.replace('_', ' ').title()}")


async def adaptive_learning_style_lesson():
    """Lesson adapted to different learning styles"""
    
    print("\n" + "="*70)
    print("Example 2: Adaptive Learning Styles")
    print("="*70)
    
    teacher = TeachingAgent()
    
    topic = "photosynthesis"
    learning_styles = ["visual", "auditory", "kinesthetic"]
    
    for style in learning_styles:
        print(f"\n{'-'*70}")
        print(f"Teaching for {style.upper()} learner")
        print(f"{'-'*70}")
        
        result = await teacher.process({
            "student_profile": {
                "student_id": f"student_{style}",
                "grade_level": "7th grade",
                "learning_style": style,
                "interests": ["science", "nature"],
                "mastery_levels": {}
            },
            "topic": topic
        })
        
        # Show excerpt
        content = result.content
        excerpt = content[:500] + "..." if len(content) > 500 else content
        print(excerpt)


async def progressive_difficulty_lesson():
    """Lesson that adapts difficulty based on mastery"""
    
    print("\n" + "="*70)
    print("Example 3: Progressive Difficulty Adaptation")
    print("="*70)
    
    teacher = TeachingAgent()
    
    # Student with different mastery levels
    mastery_levels = [
        ("beginner", 0.2),
        ("intermediate", 0.6),
        ("advanced", 0.9)
    ]
    
    topic = "fractions"
    
    for level_name, mastery in mastery_levels:
        print(f"\n{'-'*70}")
        print(f"Lesson for {level_name.upper()} student (mastery: {mastery:.0%})")
        print(f"{'-'*70}")
        
        result = await teacher.process({
            "student_profile": {
                "student_id": f"student_{level_name}",
                "grade_level": "6th grade",
                "learning_style": "balanced",
                "interests": ["math"],
                "mastery_levels": {topic: mastery}
            },
            "topic": topic
        })
        
        # Show excerpt
        excerpt = result.content[:400] + "..."
        print(excerpt)


async def multi_modal_explanation():
    """Explanation using multiple modalities"""
    
    print("\n" + "="*70)
    print("Example 4: Multi-Modal Explanation")
    print("="*70)
    
    explainer = ExplanationAgent()
    
    concept = "Newton's Third Law"
    
    print(f"\nConcept: {concept}")
    print("\nGenerating multi-modal explanation...")
    
    result = await explainer.process({
        "concept": concept,
        "level": "high_school",
        "learning_style": "kinesthetic",
        "prior_knowledge": ["Newton's First Law", "forces", "motion"]
    })
    
    print("\n" + "-"*70)
    print("Explanation")
    print("-"*70)
    print(result.content)
    
    print("\n" + "-"*70)
    print("Explanation Components")
    print("-"*70)
    components = result.metadata.get("components", {})
    for component, present in components.items():
        status = "✓" if present else "✗"
        print(f"  {status} {component.replace('_', ' ').title()}")


async def interest_based_lesson():
    """Lesson tailored to student's interests"""
    
    print("\n" + "="*70)
    print("Example 5: Interest-Based Lesson")
    print("="*70)
    
    teacher = TeachingAgent()
    
    # Student interested in video games
    result = await teacher.process({
        "student_profile": {
            "student_id": "gamer_student",
            "grade_level": "8th grade",
            "learning_style": "kinesthetic",
            "interests": ["video games", "programming", "design"],
            "mastery_levels": {"basic_programming": 0.4}
        },
        "topic": "coordinate systems",
        "question": "What are coordinate systems and why do I need them?"
    })
    
    print("\nTopic: Coordinate Systems")
    print("Student Interests: Video Games, Programming, Design")
    
    print("\n" + "-"*70)
    print("Interest-Connected Explanation")
    print("-"*70)
    print(result.content)


async def misconception_addressing_lesson():
    """Lesson that addresses common misconceptions"""
    
    print("\n" + "="*70)
    print("Example 6: Addressing Misconceptions")
    print("="*70)
    
    teacher = TeachingAgent()
    
    # Student with previous incorrect attempts
    previous_attempts = [
        {
            "question": "What is 0.5 + 0.3?",
            "answer": "0.8",
            "correct": False,
            "misconception": "Added like whole numbers"
        },
        {
            "question": "What is 0.2 × 0.4?",
            "answer": "0.8",
            "correct": False,
            "misconception": "Multiplied incorrectly"
        }
    ]
    
    print("\nStudent's Previous Attempts:")
    for attempt in previous_attempts:
        print(f"  Q: {attempt['question']}")
        print(f"  A: {attempt['answer']} (Incorrect)")
        print(f"  Misconception: {attempt['misconception']}")
    
    result = await teacher.process({
        "student_profile": {
            "student_id": "struggling_student",
            "grade_level": "5th grade",
            "learning_style": "visual",
            "interests": ["sports"],
            "mastery_levels": {"decimals": 0.3}
        },
        "topic": "decimal operations",
        "previous_attempts": previous_attempts
    })
    
    print("\n" + "-"*70)
    print("Misconception-Targeted Lesson")
    print("-"*70)
    print(result.content)


async def scaffolded_learning():
    """Lesson with scaffolding for struggling student"""
    
    print("\n" + "="*70)
    print("Example 7: Scaffolded Learning")
    print("="*70)
    
    teacher = TeachingAgent()
    knowledge_state = KnowledgeState("struggling_student")
    
    # Student struggling with current topic
    topic = "solving equations with variables on both sides"
    
    print(f"\nTopic: {topic}")
    print("Student Status: Struggling (low mastery)")
    
    result = await teacher.process({
        "student_profile": {
            "student_id": "struggling_student",
            "grade_level": "8th grade",
            "learning_style": "visual",
            "interests": ["music"],
            "mastery_levels": {
                "basic_algebra": 0.5,
                "solving_simple_equations": 0.6,
                topic: 0.2
            }
        },
        "topic": topic,
        "context": "Student needs extra support and step-by-step guidance"
    })
    
    print("\n" + "-"*70)
    print("Scaffolded Lesson (Extra Support)")
    print("-"*70)
    print(result.content)


async def main():
    """Run all personalized lesson examples"""
    
    print("\n" + "="*70)
    print("Personalized Lesson Examples")
    print("="*70)
    
    await basic_personalized_lesson()
    
    await adaptive_learning_style_lesson()
    
    await progressive_difficulty_lesson()
    
    await multi_modal_explanation()
    
    await interest_based_lesson()
    
    await misconception_addressing_lesson()
    
    await scaffolded_learning()
    
    print("\n" + "="*70)
    print("All Personalized Lesson Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())