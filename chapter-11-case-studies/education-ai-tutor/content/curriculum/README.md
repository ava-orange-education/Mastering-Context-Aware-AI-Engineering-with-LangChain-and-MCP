# Curriculum Content

This directory contains structured curriculum content for the AI tutor.

## Structure
```
curriculum/
├── math/
│   ├── algebra/
│   ├── geometry/
│   ├── calculus/
│   └── statistics/
├── science/
│   ├── physics/
│   ├── chemistry/
│   └── biology/
├── language/
│   ├── grammar/
│   └── literature/
└── history/
    ├── world_history/
    └── us_history/
```

## Content Format

Each topic should include:

- **Lesson Content**: Main teaching material
- **Prerequisites**: Required prior knowledge
- **Learning Objectives**: What students should learn
- **Difficulty Level**: Beginner, Easy, Medium, Hard, Advanced
- **Estimated Time**: Time needed to master
- **Examples**: Concrete examples
- **Common Misconceptions**: Typical errors students make

## Example Content File
```json
{
  "topic": "quadratic_equations",
  "subject": "math",
  "category": "algebra",
  "grade_level": "9-10",
  "difficulty": "medium",
  "prerequisites": ["linear_equations", "factoring"],
  "learning_objectives": [
    "Understand the standard form of quadratic equations",
    "Solve quadratic equations using multiple methods",
    "Apply quadratic equations to real-world problems"
  ],
  "content": "...",
  "examples": [...],
  "practice_problems": [...],
  "common_misconceptions": [...]
}
```

## Adding Content

1. Create a new JSON file for the topic
2. Include all required fields
3. Add examples and practice problems
4. Tag with appropriate metadata
5. Update the vector database