# Education AI Tutor

AI-powered personalized learning assistant that adapts to individual student needs and learning styles.

## Overview

This case study demonstrates a production-ready AI tutoring system that:

- Provides personalized learning experiences
- Adapts to individual learning styles and pace
- Tracks student progress and mastery
- Generates custom learning materials
- Offers real-time explanations and feedback
- Integrates with Learning Management Systems (LMS)

## Architecture
```
┌─────────────────────────────────────────────────┐
│           Student Query/Request                  │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │  Student Model        │
         │  (Learning Profile)   │
         └───────────┬───────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   ┌────────┐  ┌─────────┐  ┌──────────┐
   │Teaching│  │Practice │  │Assessment│
   │Agent   │  │Generator│  │Agent     │
   │        │  │Agent    │  │          │
   └────┬───┘  └────┬────┘  └────┬─────┘
        │           │            │
        └───────────┼────────────┘
                    │
           ┌────────┴────────┐
           │                 │
           ▼                 ▼
    ┌──────────┐      ┌──────────┐
    │Knowledge │      │Progress  │
    │Base RAG  │      │Tracker   │
    └──────────┘      └──────────┘
```

## Key Features

### Personalized Learning
- Adaptive difficulty adjustment
- Learning style recognition (visual, auditory, kinesthetic)
- Pace customization
- Interest-based content selection

### Multi-Modal Teaching
- Text explanations with examples
- Visual diagrams and illustrations
- Interactive exercises
- Step-by-step walkthroughs

### Progress Tracking
- Mastery level monitoring
- Concept dependency mapping
- Strength/weakness identification
- Learning analytics

### Content Generation
- Custom practice problems
- Tailored explanations
- Adaptive hints and scaffolding
- Multi-difficulty exercises

## Quick Start

### Prerequisites

- Python 3.9+
- Anthropic API key
- Vector database (Pinecone/Weaviate)
- Optional: LMS integration (Canvas, Moodle, etc.)

### Installation
```bash
cd education-ai-tutor

# Install dependencies
pip install -r ../requirements.txt

# Configure environment
cp ../.env.example ../.env
# Add your API keys
```

### Configuration

Required environment variables:
```bash
# Core
ANTHROPIC_API_KEY=your_key
VECTOR_STORE=pinecone

# Education-specific
EDUCATION_GRADE_LEVELS=K-12,Undergraduate,Graduate
EDUCATION_SUBJECTS=Math,Science,History,Language
```

### Running Examples

**Student Learning Session:**
```bash
python examples/01_personalized_learning_session.py
```

**Practice Problem Generation:**
```bash
python examples/02_practice_generation.py
```

**Progress Assessment:**
```bash
python examples/03_progress_assessment.py
```

## Usage

### Basic Tutoring Session
```python
from agents.teaching_agent import TeachingAgent
from student_model.learning_profile import LearningProfile

# Initialize agent
teaching_agent = TeachingAgent()

# Create student profile
student = LearningProfile(
    student_id="student_123",
    grade_level="9th",
    learning_style="visual",
    current_level={"algebra": "intermediate"}
)

# Get explanation
result = await teaching_agent.process({
    "student_profile": student,
    "topic": "quadratic equations",
    "question": "How do I solve x² + 5x + 6 = 0?"
})

print(result.content)
```

### Generate Practice Problems
```python
from agents.practice_generator_agent import PracticeGeneratorAgent

generator = PracticeGeneratorAgent()

problems = await generator.process({
    "student_profile": student,
    "topic": "quadratic equations",
    "difficulty": "medium",
    "count": 5
})

for problem in problems.metadata["problems"]:
    print(f"Problem: {problem['question']}")
    print(f"Solution: {problem['solution']}")
```

### Track Progress
```python
from agents.assessment_agent import AssessmentAgent

assessment_agent = AssessmentAgent()

result = await assessment_agent.process({
    "student_id": "student_123",
    "topic": "algebra",
    "responses": student_answers
})

print(f"Mastery Level: {result.metadata['mastery_level']}")
print(f"Strengths: {result.metadata['strengths']}")
print(f"Areas for Improvement: {result.metadata['weaknesses']}")
```

## API Endpoints

Start the API server:
```bash
uvicorn api.main:app --reload --port 8002
```

### Endpoints

**Get Explanation:**
```bash
POST /api/v1/explain
Content-Type: application/json

{
  "student_id": "student_123",
  "topic": "quadratic equations",
  "question": "How do I factor x² + 7x + 12?"
}
```

**Generate Practice:**
```bash
POST /api/v1/practice
Content-Type: application/json

{
  "student_id": "student_123",
  "topic": "algebra",
  "difficulty": "medium",
  "count": 10
}
```

**Submit Assessment:**
```bash
POST /api/v1/assess
Content-Type: application/json

{
  "student_id": "student_123",
  "topic": "algebra",
  "answers": [
    {"question_id": "q1", "answer": "x = 2, x = 3"}
  ]
}
```

**Get Progress:**
```bash
GET /api/v1/progress/{student_id}
```

## Student Model

### Learning Profile
```python
{
  "student_id": "student_123",
  "grade_level": "9th",
  "learning_style": "visual",
  "pace_preference": "moderate",
  "interests": ["science", "technology"],
  "mastery_levels": {
    "algebra": 0.75,
    "geometry": 0.60,
    "trigonometry": 0.45
  },
  "learning_goals": [
    "Master quadratic equations by end of month",
    "Improve problem-solving speed"
  ]
}
```

### Adaptive Learning

The system adapts based on:
- **Response accuracy**: Adjusts difficulty
- **Time taken**: Adjusts pace and scaffolding
- **Error patterns**: Identifies misconceptions
- **Engagement**: Modifies content presentation
- **Learning style**: Selects appropriate modalities

## Knowledge Base

### Educational Content Structure
```
content/
├── math/
│   ├── algebra/
│   │   ├── linear_equations.md
│   │   ├── quadratic_equations.md
│   │   └── systems_of_equations.md
│   ├── geometry/
│   └── calculus/
├── science/
│   ├── physics/
│   ├── chemistry/
│   └── biology/
└── language/
    ├── grammar/
    └── literature/
```

### Content Format

Each topic includes:
- Concept explanation
- Prerequisite concepts
- Difficulty progression
- Common misconceptions
- Practice problems
- Real-world applications

## Personalization Strategies

### Learning Style Adaptation

**Visual Learners:**
- Diagrams and charts
- Color-coded examples
- Spatial representations

**Auditory Learners:**
- Step-by-step verbal explanations
- Mnemonic devices
- Rhythm patterns for memorization

**Kinesthetic Learners:**
- Interactive exercises
- Hands-on examples
- Physical analogies

### Difficulty Adaptation
```python
def adjust_difficulty(student_performance):
    if accuracy > 0.8 and time_taken < expected:
        return "increase_difficulty"
    elif accuracy < 0.5:
        return "decrease_difficulty"
    else:
        return "maintain_difficulty"
```

## Progress Tracking

### Mastery Metrics

- **Concept Mastery**: 0.0 to 1.0 scale
- **Retention Rate**: Long-term knowledge retention
- **Problem-Solving Speed**: Efficiency improvement
- **Error Reduction**: Mistake pattern analysis

### Learning Analytics
```python
{
  "total_sessions": 45,
  "total_time_spent": "23 hours",
  "topics_mastered": 12,
  "topics_in_progress": 5,
  "average_mastery_gain": 0.15,
  "streak_days": 7,
  "achievements": ["Perfect Week", "Fast Learner"]
}
```

## Evaluation

### Teaching Effectiveness
```python
from evaluation.teaching_effectiveness import TeachingEffectivenessEvaluator

evaluator = TeachingEffectivenessEvaluator()

metrics = evaluator.evaluate(
    student_sessions=sessions,
    pre_assessments=pre_tests,
    post_assessments=post_tests
)

print(f"Learning Gain: {metrics['learning_gain']:.2%}")
print(f"Engagement Score: {metrics['engagement_score']:.2f}")
```

### Student Satisfaction

- Explanation clarity ratings
- Difficulty appropriateness
- Learning pace satisfaction
- Overall experience feedback

## LMS Integration

### Canvas Integration
```python
from integrations.canvas_connector import CanvasConnector

canvas = CanvasConnector()
await canvas.initialize()

# Sync grades
grades = await canvas.get_student_grades(student_id)

# Submit assignment
await canvas.submit_assignment(
    course_id="101",
    assignment_id="501",
    student_id=student_id,
    submission=submission_data
)
```

### Moodle Integration
```python
from integrations.moodle_connector import MoodleConnector

moodle = MoodleConnector()

# Get course content
content = await moodle.get_course_content(course_id)

# Track progress
await moodle.update_progress(student_id, activity_id, status)
```

## Gamification

### Achievement System

- **Streaks**: Daily learning streaks
- **Mastery Badges**: Topic completion badges
- **Speed Bonuses**: Fast problem solving
- **Improvement Awards**: Progress milestones

### Leaderboards

- Class rankings (opt-in)
- Personal bests
- Challenge competitions

## Content Coverage

### Mathematics
- Arithmetic (K-6)
- Pre-Algebra (6-8)
- Algebra I & II (8-10)
- Geometry (9-10)
- Trigonometry (10-11)
- Pre-Calculus (11-12)
- Calculus (12+)
- Statistics

### Science
- General Science (K-8)
- Biology
- Chemistry
- Physics
- Earth Science

### Language Arts
- Reading Comprehension
- Grammar & Writing
- Literature Analysis
- Vocabulary Building

## Testing
```bash
# Run all tests
pytest tests/

# Test specific modules
pytest tests/test_agents.py
pytest tests/test_student_model.py
pytest tests/test_personalization.py
```

## Production Deployment

See `docs/deployment_guide.md` for:
- Scaling for multiple students
- Database setup for student data
- Content delivery optimization
- Privacy and data protection
- COPPA/FERPA compliance

## Privacy & Security

- Student data encryption
- COPPA compliance (under 13)
- FERPA compliance (education records)
- Parental consent management
- Data retention policies

## Limitations

- Requires structured curriculum content
- Best for STEM subjects (current version)
- Limited support for creative subjects
- Assumes basic digital literacy

## Future Enhancements

- Multi-language support
- Collaborative learning features
- Parent/teacher dashboard
- AR/VR integration
- Voice-based interaction

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: GitHub Issues

## License

Educational software - see LICENSE for details.