"""
Example 3: Progress Tracking

Demonstrates progress tracking and learning analytics
"""

import asyncio
import sys
sys.path.append('../..')

from student_model.knowledge_state import KnowledgeState
from student_model.progress_tracker import ProgressTracker
from student_model.learning_style import LearningStyleAnalyzer
from personalization.learning_path_generator import LearningPathGenerator
from evaluation.learning_outcomes import LearningOutcomesEvaluator
from evaluation.engagement_metrics import EngagementMetrics
from shared.utils import setup_logging

setup_logging()


async def knowledge_state_tracking():
    """Track student knowledge state over time"""
    
    print("\n" + "="*70)
    print("Example 1: Knowledge State Tracking")
    print("="*70)
    
    knowledge_state = KnowledgeState("student_001")
    
    # Simulate learning sessions
    sessions = [
        ("algebra_basics", True, "easy", 45, 1),
        ("algebra_basics", True, "medium", 50, 1),
        ("linear_equations", False, "medium", 90, 2),
        ("linear_equations", True, "medium", 70, 1),
        ("linear_equations", True, "medium", 60, 1),
        ("quadratic_equations", False, "hard", 120, 3),
    ]
    
    print("\nSimulating Learning Sessions:")
    for topic, correct, difficulty, time, attempts in sessions:
        knowledge_state.update_mastery(
            concept=topic,
            performance={
                "correct": correct,
                "difficulty": difficulty,
                "time_taken": time,
                "attempts": attempts
            }
        )
        
        status = "✓" if correct else "✗"
        print(f"  {status} {topic} ({difficulty}, {time}s, {attempts} attempt(s))")
    
    # Show knowledge map
    print("\n" + "-"*70)
    print("Knowledge Map")
    print("-"*70)
    
    knowledge_map = knowledge_state.get_knowledge_map()
    
    print(f"\nTotal Concepts: {knowledge_map['total_concepts']}")
    print(f"Mastered: {knowledge_map['mastered_concepts']}")
    print(f"In Progress: {knowledge_map['in_progress']}")
    print(f"Needs Practice: {knowledge_map['needs_practice']}")
    
    print("\nMastery Levels:")
    for concept, mastery in knowledge_map['mastery_levels'].items():
        bar_length = int(mastery * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {concept}: {bar} {mastery:.0%}")


async def progress_over_time():
    """Track progress over time"""
    
    print("\n" + "="*70)
    print("Example 2: Progress Over Time")
    print("="*70)
    
    tracker = ProgressTracker("student_002")
    
    # Record multiple sessions
    print("\nRecording Learning Sessions:")
    
    sessions_data = [
        ("Mathematics", "Algebra", 30, 10, 8),
        ("Mathematics", "Geometry", 45, 12, 10),
        ("Science", "Physics", 40, 8, 7),
        ("Mathematics", "Algebra", 35, 10, 9),
        ("Science", "Chemistry", 50, 15, 12),
    ]
    
    for subject, topic, duration, attempted, correct in sessions_data:
        tracker.record_session(
            subject=subject,
            topic=topic,
            duration=duration,
            problems_attempted=attempted,
            problems_correct=correct,
            concepts_covered=[topic],
            difficulty_level="medium"
        )
        
        accuracy = correct / attempted if attempted > 0 else 0
        print(f"  {subject}/{topic}: {duration}min, {accuracy:.0%} accuracy")
    
    # Get progress summary
    print("\n" + "-"*70)
    print("Progress Summary (Last 30 Days)")
    print("-"*70)
    
    summary = tracker.get_progress_summary(days=30)
    
    print(f"\nTotal Sessions: {summary['total_sessions']}")
    print(f"Total Time: {summary['total_time_hours']:.1f} hours")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.0%}")
    print(f"Current Streak: {summary['current_streak']} days")
    print(f"Longest Streak: {summary['longest_streak']} days")
    
    if summary.get('accuracy_by_subject'):
        print("\nAccuracy by Subject:")
        for subject, accuracy in summary['accuracy_by_subject'].items():
            print(f"  {subject}: {accuracy:.0%}")


async def learning_style_analysis():
    """Analyze and adapt to learning style"""
    
    print("\n" + "="*70)
    print("Example 3: Learning Style Analysis")
    print("="*70)
    
    analyzer = LearningStyleAnalyzer("student_003")
    
    # Record interactions with different content types
    print("\nRecording Content Interactions:")
    
    interactions = [
        ("diagram", "visual", 0.9, 300, True),
        ("video", "visual", 0.85, 600, True),
        ("explanation", "auditory", 0.6, 400, True),
        ("interactive", "kinesthetic", 0.95, 900, True),
        ("diagram", "visual", 0.9, 280, True),
        ("explanation", "auditory", 0.7, 420, False),
        ("hands_on", "kinesthetic", 0.95, 1200, True),
    ]
    
    for content_type, style, effectiveness, time_spent, completed in interactions:
        analyzer.record_interaction(
            content_type=content_type,
            learning_style=style,
            effectiveness=effectiveness,
            engagement_metrics={
                "time_spent": time_spent,
                "completion": completed,
                "revisited": False
            }
        )
        
        print(f"  {content_type} ({style}): {effectiveness:.0%} effective")
    
    # Get style analysis
    print("\n" + "-"*70)
    print("Learning Style Analysis")
    print("-"*70)
    
    dominant = analyzer.get_dominant_style()
    distribution = analyzer.get_style_distribution()
    
    print(f"\nDominant Style: {dominant.title()}")
    print("\nStyle Distribution:")
    for style, score in distribution.items():
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {style.title():12} {bar} {score:.0%}")
    
    # Get recommendations
    recommendations = analyzer.get_personalization_recommendations()
    
    print("\n" + "-"*70)
    print("Content Recommendations")
    print("-"*70)
    for suggestion in recommendations.get("content_suggestions", []):
        print(f"  • {suggestion}")


async def learning_path_progress():
    """Track progress along learning path"""
    
    print("\n" + "="*70)
    print("Example 4: Learning Path Progress")
    print("="*70)
    
    path_generator = LearningPathGenerator()
    knowledge_state = KnowledgeState("student_004")
    
    # Set up concept dependencies
    path_generator.add_concept_dependency(
        "addition", [], 30, "beginner"
    )
    path_generator.add_concept_dependency(
        "subtraction", ["addition"], 35, "beginner"
    )
    path_generator.add_concept_dependency(
        "multiplication", ["addition"], 45, "easy"
    )
    path_generator.add_concept_dependency(
        "division", ["multiplication"], 50, "easy"
    )
    path_generator.add_concept_dependency(
        "fractions", ["multiplication", "division"], 60, "medium"
    )
    
    # Current knowledge
    knowledge_state.mastery_levels = {
        "addition": 0.9,
        "subtraction": 0.85,
        "multiplication": 0.7,
        "division": 0.5
    }
    
    print("\nCurrent Mastery:")
    for concept, mastery in knowledge_state.mastery_levels.items():
        print(f"  {concept}: {mastery:.0%}")
    
    # Generate learning path to goal
    print("\n" + "-"*70)
    print("Learning Path to Master Fractions")
    print("-"*70)
    
    path = path_generator.generate_learning_path(
        student_id="student_004",
        learning_goal="fractions",
        current_knowledge=knowledge_state.mastery_levels,
        target_mastery=0.8
    )
    
    print(f"\nGoal: {path['learning_goal']}")
    print(f"Estimated Time: {path['estimated_time_minutes']} minutes ({path['estimated_days']} days)")
    print(f"Concepts to Learn: {path['concepts_to_learn']}")
    
    print("\nLearning Path:")
    for i, step in enumerate(path['path'], 1):
        current_mastery = step['current_mastery']
        status = "✓" if current_mastery >= 0.8 else "→" if current_mastery > 0 else "○"
        print(f"  {i}. {status} {step['concept']} ({current_mastery:.0%} → 80%)")
        print(f"     Estimated time: {step['estimated_time']} min")
    
    if path['milestones']:
        print("\nMilestones:")
        for milestone in path['milestones']:
            print(f"  {milestone['milestone_number']}. {milestone['concept']}")
            print(f"     Progress: {milestone['progress_percentage']:.0f}%")


async def learning_outcomes_evaluation():
    """Evaluate learning outcomes"""
    
    print("\n" + "="*70)
    print("Example 5: Learning Outcomes Evaluation")
    print("="*70)
    
    evaluator = LearningOutcomesEvaluator()
    
    # Record pre and post assessments
    topic = "quadratic_equations"
    
    evaluator.record_pre_assessment(
        student_id="student_005",
        topic=topic,
        assessment_results={
            "score": 0.4,
            "mastery_level": 0.35
        }
    )
    
    print(f"\nPre-Assessment: {topic}")
    print("  Score: 40%")
    print("  Mastery: 35%")
    
    # Simulate learning period
    print("\n[Student studies for 2 weeks...]")
    
    evaluator.record_post_assessment(
        student_id="student_005",
        topic=topic,
        assessment_results={
            "score": 0.85,
            "mastery_level": 0.8
        }
    )
    
    print("\nPost-Assessment:")
    print("  Score: 85%")
    print("  Mastery: 80%")
    
    # Calculate learning gain
    print("\n" + "-"*70)
    print("Learning Gain Analysis")
    print("-"*70)
    
    gain = evaluator.calculate_learning_gain("student_005", topic)
    
    print(f"\nAbsolute Gain: {gain['absolute_gain']:.0%}")
    print(f"Normalized Gain: {gain['normalized_gain']:.2f}")
    print(f"Gain Level: {gain['gain_level'].upper()}")
    print(f"Mastery Improvement: {gain['mastery_gain']:.0%}")


async def engagement_tracking():
    """Track student engagement"""
    
    print("\n" + "="*70)
    print("Example 6: Engagement Tracking")
    print("="*70)
    
    metrics = EngagementMetrics()
    
    # Record sessions
    print("\nRecording Learning Sessions:")
    
    sessions_data = [
        (45, ["algebra"], 10, 9, True),
        (60, ["geometry"], 12, 10, True),
        (30, ["algebra"], 8, 6, True),
        (50, ["trigonometry"], 10, 8, False),
        (40, ["algebra"], 12, 11, True),
    ]
    
    for duration, topics, attempted, completed, voluntary in sessions_data:
        metrics.record_session(
            student_id="student_006",
            duration=duration,
            topics_covered=topics,
            problems_attempted=attempted,
            problems_completed=completed,
            voluntary=voluntary
        )
        
        vol_status = "✓" if voluntary else "✗"
        print(f"  {topics[0]}: {duration}min, {completed}/{attempted} problems {vol_status}")
    
    # Calculate engagement score
    print("\n" + "-"*70)
    print("Engagement Analysis")
    print("-"*70)
    
    engagement = metrics.calculate_engagement_score("student_006", days=7)
    
    print(f"\nEngagement Score: {engagement['engagement_score']:.0%}")
    print(f"Level: {engagement['level'].replace('_', ' ').title()}")
    print(f"Total Sessions: {engagement['sessions']}")
    print(f"Total Time: {engagement['total_time']:.0f} minutes")
    print(f"Avg Session Length: {engagement['avg_session_length']:.0f} minutes")
    
    print("\nComponent Scores:")
    for component, score in engagement['component_scores'].items():
        print(f"  {component.title()}: {score:.0%}")


async def cohort_analytics():
    """Analyze cohort-level outcomes"""
    
    print("\n" + "="*70)
    print("Example 7: Cohort Analytics")
    print("="*70)
    
    evaluator = LearningOutcomesEvaluator()
    
    # Simulate cohort data
    topic = "algebra_basics"
    students = [
        ("student_A", 0.3, 0.85),
        ("student_B", 0.5, 0.90),
        ("student_C", 0.2, 0.70),
        ("student_D", 0.4, 0.88),
        ("student_E", 0.6, 0.95),
    ]
    
    print(f"\nCohort Learning: {topic}")
    print("\nStudent Pre → Post Scores:")
    
    for student_id, pre_score, post_score in students:
        evaluator.record_pre_assessment(
            student_id=student_id,
            topic=topic,
            assessment_results={"score": pre_score, "mastery_level": pre_score}
        )
        
        evaluator.record_post_assessment(
            student_id=student_id,
            topic=topic,
            assessment_results={"score": post_score, "mastery_level": post_score}
        )
        
        print(f"  {student_id}: {pre_score:.0%} → {post_score:.0%}")
    
    # Evaluate cohort outcomes
    print("\n" + "-"*70)
    print("Cohort Outcomes")
    print("-"*70)
    
    outcomes = evaluator.evaluate_cohort_outcomes(topic)
    
    print(f"\nStudents Evaluated: {outcomes['students_evaluated']}")
    print(f"Mean Gain: {outcomes['mean_absolute_gain']:.0%}")
    print(f"Median Gain: {outcomes['median_absolute_gain']:.0%}")
    print(f"Mean Normalized Gain: {outcomes['mean_normalized_gain']:.2f}")
    
    print("\nGain Distribution:")
    print(f"  High Gain: {outcomes['high_gain_count']} students")
    print(f"  Medium Gain: {outcomes['medium_gain_count']} students")
    print(f"  Low Gain: {outcomes['low_gain_count']} students")


async def main():
    """Run all progress tracking examples"""
    
    print("\n" + "="*70)
    print("Progress Tracking Examples")
    print("="*70)
    
    await knowledge_state_tracking()
    
    await progress_over_time()
    
    await learning_style_analysis()
    
    await learning_path_progress()
    
    await learning_outcomes_evaluation()
    
    await engagement_tracking()
    
    await cohort_analytics()
    
    print("\n" + "="*70)
    print("All Progress Tracking Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())