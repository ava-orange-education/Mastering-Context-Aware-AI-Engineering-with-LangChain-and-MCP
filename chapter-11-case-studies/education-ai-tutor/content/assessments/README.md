# Assessment Content

This directory contains assessment materials for evaluating student knowledge.

## Types of Assessments

### Pre-Assessments
- Determine baseline knowledge
- Identify prerequisites gaps
- Guide initial difficulty level

### Formative Assessments
- Ongoing checks during learning
- Provide immediate feedback
- Guide instructional adjustments

### Summative Assessments
- Evaluate mastery after learning
- Measure achievement of objectives
- Determine readiness to progress

## Assessment Format
```json
{
  "assessment_id": "algebra_unit_1",
  "type": "formative",
  "topic": "linear_equations",
  "difficulty": "medium",
  "questions": [
    {
      "question_id": "q1",
      "question": "Solve for x: 2x + 5 = 13",
      "type": "short_answer",
      "correct_answer": "x = 4",
      "solution_steps": [...],
      "common_errors": [...],
      "hints": [...]
    }
  ],
  "rubric": {...},
  "time_limit": 30
}
```

## Question Types

- **Multiple Choice**: Select one correct answer
- **Multi-Select**: Select all correct answers
- **Short Answer**: Brief written response
- **Essay**: Extended written response
- **Problem Solving**: Show work and solution
- **True/False**: Binary choice

## Creating Assessments

1. Align with learning objectives
2. Match appropriate difficulty level
3. Include variety of question types
4. Provide clear rubrics
5. Add hints and scaffolding
6. Include common misconceptions