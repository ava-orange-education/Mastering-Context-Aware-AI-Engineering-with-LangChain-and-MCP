# Evaluation Frameworks for Context-Aware AI Systems

## Overview

This document provides comprehensive evaluation frameworks for assessing the performance, quality, and impact of context-aware AI systems across all four domains.

## Evaluation Dimensions

### 1. Technical Performance
- Response latency
- Throughput
- Resource utilization
- System availability

### 2. AI Quality
- Accuracy
- Relevance
- Consistency
- Hallucination rate

### 3. User Experience
- Task completion rate
- Time to complete task
- User satisfaction
- Engagement metrics

### 4. Business Impact
- ROI
- Productivity gains
- Error reduction
- Cost savings

### 5. Safety and Compliance
- Regulatory compliance
- Safety incidents
- Security events
- Audit findings

## Healthcare DNA Wellness Evaluation

### Medical Accuracy Metrics

#### 1. Variant Interpretation Accuracy
```python
def evaluate_variant_interpretation(predictions, ground_truth):
    """
    Evaluate DNA variant interpretation against ClinVar gold standard
    """
    metrics = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'precision': precision_score(ground_truth, predictions),
        'recall': recall_score(ground_truth, predictions),
        'f1': f1_score(ground_truth, predictions)
    }
    
    # Class-wise metrics
    for classification in ['pathogenic', 'likely_pathogenic', 'benign', 'vus']:
        metrics[f'{classification}_precision'] = class_precision(classification)
        metrics[f'{classification}_recall'] = class_recall(classification)
    
    return metrics
```

**Benchmarks:**
- Accuracy: >95% for pathogenic variants
- Recall: >98% for clinically significant variants
- False positive rate: <2%

#### 2. Clinical Guideline Relevance
```python
def evaluate_guideline_relevance(recommendations, clinical_context):
    """
    Evaluate if recommendations match appropriate clinical guidelines
    """
    # LLM-as-judge evaluation
    prompt = f"""
    Clinical Context: {clinical_context}
    Recommendation: {recommendations}
    
    Evaluate:
    1. Does recommendation follow evidence-based guidelines?
    2. Is evidence level appropriate (A, B, C)?
    3. Are contraindications considered?
    4. Is patient context addressed?
    
    Score 1-5 for each dimension.
    """
    
    scores = llm_evaluate(prompt)
    return scores
```

**Target Scores:**
- Guideline adherence: >4.5/5
- Evidence quality: Level A or B
- Contraindication detection: 100%

#### 3. Safety Validation
```python
def safety_evaluation(recommendations):
    """
    Check for potentially harmful recommendations
    """
    safety_checks = {
        'drug_interactions': check_drug_interactions(recommendations),
        'contraindications': check_contraindications(recommendations),
        'dosage_safety': validate_dosages(recommendations),
        'allergy_check': check_allergies(recommendations)
    }
    
    # Zero tolerance for safety issues
    assert all(safety_checks.values()), "Safety check failed!"
    
    return safety_checks
```

**Requirements:**
- Zero harmful recommendations
- 100% contraindication detection
- All drug interactions flagged

### HIPAA Compliance Metrics
```python
def hipaa_compliance_audit():
    """
    Audit HIPAA compliance
    """
    checks = {
        'encryption_at_rest': verify_encryption(),
        'encryption_in_transit': verify_tls(),
        'access_logs_complete': verify_audit_logs(),
        'minimum_necessary': verify_access_scope(),
        'phi_redaction': verify_redaction()
    }
    
    compliance_rate = sum(checks.values()) / len(checks)
    return compliance_rate, checks
```

**Target:** 100% compliance across all checks

### Patient Outcomes

- **Preventive Care Adoption**: % of recommendations followed
- **Health Metric Improvements**: Changes in measurable health indicators
- **Patient Satisfaction**: Survey scores (1-5 scale)
- **Clinical Validation**: Physician review of recommendations

**Measurement Period:** 6-12 months for outcome validation

## Enterprise Knowledge Assistant Evaluation

### Search Quality Metrics

#### 1. Retrieval Metrics
```python
def evaluate_retrieval(queries, relevance_judgments):
    """
    Evaluate search retrieval quality
    """
    metrics = {}
    
    for k in [1, 5, 10]:
        metrics[f'precision@{k}'] = precision_at_k(queries, relevance_judgments, k)
        metrics[f'recall@{k}'] = recall_at_k(queries, relevance_judgments, k)
    
    metrics['ndcg'] = ndcg_score(queries, relevance_judgments)
    metrics['mrr'] = mean_reciprocal_rank(queries, relevance_judgments)
    metrics['map'] = mean_average_precision(queries, relevance_judgments)
    
    return metrics
```

**Benchmarks:**
- Precision@5: >0.8
- NDCG: >0.85
- MRR: >0.7

#### 2. Answer Quality
```python
def evaluate_answer_quality(questions, generated_answers, reference_answers):
    """
    Evaluate generated answer quality
    """
    # Semantic similarity
    similarity_scores = []
    for gen, ref in zip(generated_answers, reference_answers):
        similarity = semantic_similarity(gen, ref)
        similarity_scores.append(similarity)
    
    # Factual accuracy
    factuality_scores = []
    for gen, ref in zip(generated_answers, reference_answers):
        factuality = check_factual_consistency(gen, ref)
        factuality_scores.append(factuality)
    
    # Completeness
    completeness_scores = []
    for gen, ref in zip(generated_answers, reference_answers):
        completeness = check_completeness(gen, ref)
        completeness_scores.append(completeness)
    
    return {
        'avg_similarity': np.mean(similarity_scores),
        'avg_factuality': np.mean(factuality_scores),
        'avg_completeness': np.mean(completeness_scores)
    }
```

**Targets:**
- Semantic similarity: >0.85
- Factual accuracy: >95%
- Completeness: >80%

### Permission Enforcement
```python
def test_permission_enforcement():
    """
    Verify permission-aware retrieval
    """
    test_cases = [
        {
            'user': 'user_a',
            'query': 'confidential project',
            'should_see': ['doc_1', 'doc_2'],
            'should_not_see': ['doc_3', 'doc_4']
        },
        # ... more test cases
    ]
    
    violations = 0
    for case in test_cases:
        results = search(case['query'], user=case['user'])
        result_ids = [r.id for r in results]
        
        # Check for unauthorized access
        for forbidden_doc in case['should_not_see']:
            if forbidden_doc in result_ids:
                violations += 1
        
        # Check for missing authorized docs
        for allowed_doc in case['should_see']:
            if allowed_doc not in result_ids:
                violations += 1
    
    accuracy = 1 - (violations / total_checks)
    return accuracy
```

**Requirement:** 100% permission enforcement (zero unauthorized access)

### User Productivity

**Metrics:**
- Time saved per search (vs. manual search)
- Documents found per session
- Search success rate
- Repeat search rate (lower is better)

**Measurement:**
```python
def calculate_productivity_gains(usage_data):
    """
    Calculate productivity improvements
    """
    # Time saved
    avg_manual_search_time = 15  # minutes
    avg_ai_search_time = usage_data['avg_session_duration']
    time_saved = (avg_manual_search_time - avg_ai_search_time) * usage_data['search_count']
    
    # Search success rate
    success_rate = usage_data['found_answer'] / usage_data['total_searches']
    
    # ROI
    hourly_rate = 50  # dollars
    time_saved_hours = time_saved / 60
    value_generated = time_saved_hours * hourly_rate
    
    return {
        'time_saved_hours': time_saved_hours,
        'value_generated': value_generated,
        'success_rate': success_rate
    }
```

**Targets:**
- Time saved: >10 minutes per search
- Success rate: >85%
- ROI: >300%

## Education AI Tutor Evaluation

### Learning Outcomes

#### 1. Knowledge Gain
```python
def evaluate_learning_gains(students):
    """
    Measure learning effectiveness
    """
    gains = []
    
    for student in students:
        pre_test = student.pre_test_score
        post_test = student.post_test_score
        
        # Normalized gain (Hake gain)
        if pre_test < 100:
            gain = (post_test - pre_test) / (100 - pre_test)
        else:
            gain = 0
        
        gains.append(gain)
    
    return {
        'avg_gain': np.mean(gains),
        'high_gain_pct': sum(g > 0.5 for g in gains) / len(gains),
        'medium_gain_pct': sum(0.3 < g <= 0.5 for g in gains) / len(gains),
        'low_gain_pct': sum(g <= 0.3 for g in gains) / len(gains)
    }
```

**Benchmarks:**
- Average gain: >0.5 (high gain)
- High gain students: >60%
- Low gain students: <10%

#### 2. Retention
```python
def evaluate_retention(students, delay_days=30):
    """
    Measure knowledge retention after delay
    """
    retention_scores = []
    
    for student in students:
        initial_mastery = student.mastery_at_completion
        delayed_test = student.get_test_score(delay_days=delay_days)
        
        retention = delayed_test / initial_mastery
        retention_scores.append(retention)
    
    return {
        'avg_retention': np.mean(retention_scores),
        'strong_retention_pct': sum(r > 0.8 for r in retention_scores) / len(retention_scores)
    }
```

**Targets:**
- 30-day retention: >75%
- Strong retention (>80%): >50% of students

### Engagement Metrics
```python
def calculate_engagement(student_data):
    """
    Multi-dimensional engagement scoring
    """
    # Frequency: sessions per week
    frequency_score = min(student_data['sessions_per_week'] / 5, 1.0)
    
    # Duration: time per session
    duration_score = min(student_data['avg_session_minutes'] / 30, 1.0)
    
    # Completion: finished exercises
    completion_score = student_data['completed_exercises'] / student_data['assigned_exercises']
    
    # Interaction: questions asked, hints used
    interaction_score = min(student_data['interactions_per_session'] / 10, 1.0)
    
    # Voluntary: non-required practice
    voluntary_score = student_data['voluntary_practice'] / max(student_data['total_practice'], 1)
    
    # Weighted average
    engagement = (
        frequency_score * 0.2 +
        duration_score * 0.2 +
        completion_score * 0.3 +
        interaction_score * 0.15 +
        voluntary_score * 0.15
    )
    
    return engagement
```

**Targets:**
- Overall engagement: >0.7
- Completion rate: >80%
- Voluntary practice: >20%

### Pedagogical Quality
```python
def evaluate_pedagogical_quality(lessons):
    """
    Evaluate teaching quality using LLM-as-judge
    """
    eval_prompt = """
    Evaluate this lesson on:
    1. Explanation clarity (1-5)
    2. Examples quality (1-5)
    3. Difficulty appropriateness (1-5)
    4. Scaffolding effectiveness (1-5)
    5. Engagement potential (1-5)
    
    Lesson: {lesson_content}
    Student level: {student_level}
    """
    
    scores = []
    for lesson in lessons:
        score = llm_evaluate(eval_prompt.format(
            lesson_content=lesson.content,
            student_level=lesson.target_level
        ))
        scores.append(score)
    
    return {
        'avg_overall': np.mean([s['overall'] for s in scores]),
        'clarity': np.mean([s['clarity'] for s in scores]),
        'examples': np.mean([s['examples'] for s in scores]),
        'difficulty': np.mean([s['difficulty'] for s in scores]),
        'scaffolding': np.mean([s['scaffolding'] for s in scores]),
        'engagement': np.mean([s['engagement'] for s in scores])
    }
```

**Targets:**
- All dimensions: >4.0/5
- Overall quality: >4.2/5

## DevOps Monitoring Agent Evaluation

### Incident Detection Metrics
```python
def evaluate_incident_detection(ground_truth, predictions):
    """
    Evaluate incident detection accuracy
    """
    # Confusion matrix
    tp = sum((gt == 1) and (pred == 1) for gt, pred in zip(ground_truth, predictions))
    fp = sum((gt == 0) and (pred == 1) for gt, pred in zip(ground_truth, predictions))
    tn = sum((gt == 0) and (pred == 0) for gt, pred in zip(ground_truth, predictions))
    fn = sum((gt == 1) and (pred == 0) for gt, pred in zip(ground_truth, predictions))
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fpr,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }
```

**Targets:**
- Recall (detection rate): >95%
- Precision: >80%
- False positive rate: <10%

### MTTR (Mean Time To Resolve)
```python
def calculate_mttr_metrics(incidents):
    """
    Calculate MTTR and related metrics
    """
    # MTTR: Mean Time To Resolve
    resolution_times = [
        (inc.resolved_at - inc.detected_at).total_seconds()
        for inc in incidents if inc.resolved_at
    ]
    mttr = np.mean(resolution_times) if resolution_times else None
    
    # MTTA: Mean Time To Acknowledge
    ack_times = [
        (inc.acknowledged_at - inc.detected_at).total_seconds()
        for inc in incidents if inc.acknowledged_at
    ]
    mtta = np.mean(ack_times) if ack_times else None
    
    # MTTD: Mean Time To Detect
    detection_times = [
        inc.detection_delay
        for inc in incidents if inc.detection_delay
    ]
    mttd = np.mean(detection_times) if detection_times else None
    
    return {
        'mttr_seconds': mttr,
        'mttr_minutes': mttr / 60 if mttr else None,
        'mtta_seconds': mtta,
        'mttd_seconds': mttd,
        'p95_resolution_time': np.percentile(resolution_times, 95) if resolution_times else None,
        'p99_resolution_time': np.percentile(resolution_times, 99) if resolution_times else None
    }
```

**Targets by Severity:**
- Critical: MTTR <15 minutes
- High: MTTR <1 hour
- Medium: MTTR <4 hours
- Low: MTTR <24 hours

### Remediation Effectiveness
```python
def evaluate_remediation_effectiveness(actions):
    """
    Evaluate automated remediation success
    """
    # Success rate
    successful = sum(1 for a in actions if a.outcome == 'success')
    success_rate = successful / len(actions) if actions else 0
    
    # Safety: rollbacks needed
    rollbacks = sum(1 for a in actions if a.rolled_back)
    rollback_rate = rollbacks / len(actions) if actions else 0
    
    # Impact: incidents resolved
    incidents_resolved = sum(1 for a in actions if a.resolved_incident)
    resolution_rate = incidents_resolved / len(actions) if actions else 0
    
    # Time saved vs manual
    automated_time = sum(a.execution_time for a in actions)
    estimated_manual_time = len(actions) * 900  # 15 min per action
    time_saved = estimated_manual_time - automated_time
    
    return {
        'success_rate': success_rate,
        'rollback_rate': rollback_rate,
        'resolution_rate': resolution_rate,
        'time_saved_seconds': time_saved,
        'automation_benefit': time_saved / estimated_manual_time if estimated_manual_time > 0 else 0
    }
```

**Targets:**
- Success rate: >90%
- Rollback rate: <5%
- Resolution rate: >80%
- Time saved: >70%

### Decision Quality
```python
def evaluate_decision_quality(decisions):
    """
    Evaluate quality of automated decisions
    """
    # Accuracy: correct decisions
    correct = sum(1 for d in decisions if d.was_correct)
    accuracy = correct / len(decisions) if decisions else 0
    
    # Confidence calibration
    calibration_buckets = {
        'low': {'predictions': [], 'actuals': []},
        'medium': {'predictions': [], 'actuals': []},
        'high': {'predictions': [], 'actuals': []}
    }
    
    for d in decisions:
        if d.confidence < 0.4:
            bucket = 'low'
        elif d.confidence < 0.7:
            bucket = 'medium'
        else:
            bucket = 'high'
        
        calibration_buckets[bucket]['predictions'].append(d.confidence)
        calibration_buckets[bucket]['actuals'].append(1 if d.was_correct else 0)
    
    # Expected vs actual accuracy by bucket
    calibration_scores = {}
    for bucket, data in calibration_buckets.items():
        if data['actuals']:
            expected = np.mean(data['predictions'])
            actual = np.mean(data['actuals'])
            calibration_scores[bucket] = {
                'expected': expected,
                'actual': actual,
                'calibration_error': abs(expected - actual)
            }
    
    return {
        'accuracy': accuracy,
        'calibration': calibration_scores,
        'total_decisions': len(decisions)
    }
```

**Targets:**
- Decision accuracy: >85%
- Calibration error: <0.1 per bucket

## Cross-Domain Evaluation

### System Performance
```python
def evaluate_system_performance(metrics):
    """
    Standard performance metrics across all systems
    """
    return {
        'latency_p50': np.percentile(metrics['latency'], 50),
        'latency_p95': np.percentile(metrics['latency'], 95),
        'latency_p99': np.percentile(metrics['latency'], 99),
        'throughput_rps': metrics['request_count'] / metrics['time_window'],
        'error_rate': metrics['errors'] / metrics['total_requests'],
        'availability': metrics['uptime'] / metrics['total_time']
    }
```

**Universal Targets:**
- P95 latency: <500ms
- P99 latency: <1000ms
- Error rate: <0.1%
- Availability: >99.9%

### User Satisfaction
```python
def measure_user_satisfaction(feedback):
    """
    Aggregate user satisfaction metrics
    """
    # NPS: Net Promoter Score
    promoters = sum(1 for f in feedback if f.rating >= 9)
    detractors = sum(1 for f in feedback if f.rating <= 6)
    nps = (promoters - detractors) / len(feedback) * 100 if feedback else 0
    
    # CSAT: Customer Satisfaction
    satisfied = sum(1 for f in feedback if f.rating >= 7)
    csat = satisfied / len(feedback) * 100 if feedback else 0
    
    # Sentiment analysis
    sentiments = [analyze_sentiment(f.comment) for f in feedback if f.comment]
    avg_sentiment = np.mean([s.score for s in sentiments]) if sentiments else 0
    
    return {
        'nps': nps,
        'csat': csat,
        'avg_rating': np.mean([f.rating for f in feedback]),
        'avg_sentiment': avg_sentiment,
        'response_count': len(feedback)
    }
```

**Targets:**
- NPS: >50
- CSAT: >80%
- Avg rating: >4.0/5

## Continuous Evaluation

### Automated Testing Pipeline
```yaml
# .github/workflows/evaluation.yml
name: Continuous Evaluation

on:
  schedule:
    - cron: '0 0 * * *'  # Daily
  workflow_dispatch:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run Healthcare Evaluation
        run: python healthcare-dna-wellness/evaluation/run_eval.py
      
      - name: Run Enterprise Evaluation
        run: python enterprise-knowledge-assistant/evaluation/run_eval.py
      
      - name: Run Education Evaluation
        run: python education-ai-tutor/evaluation/run_eval.py
      
      - name: Run DevOps Evaluation
        run: python devops-monitoring-agent/evaluation/run_eval.py
      
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: results/
```

### Alerting on Metric Degradation
```python
def check_metric_degradation(current_metrics, baseline_metrics, thresholds):
    """
    Alert if metrics degrade beyond threshold
    """
    alerts = []
    
    for metric_name, current_value in current_metrics.items():
        baseline_value = baseline_metrics.get(metric_name)
        threshold = thresholds.get(metric_name, 0.1)  # 10% default
        
        if baseline_value:
            change = (current_value - baseline_value) / baseline_value
            
            if abs(change) > threshold:
                alerts.append({
                    'metric': metric_name,
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_pct': change * 100,
                    'threshold_pct': threshold * 100
                })
    
    return alerts
```

## Conclusion

Effective evaluation requires:

1. **Multi-Dimensional Metrics**: Technical, user experience, and business impact
2. **Domain-Specific Criteria**: Healthcare accuracy, education learning gains, etc.
3. **Continuous Monitoring**: Automated daily evaluation
4. **User Feedback Integration**: Qualitative and quantitative
5. **Regression Detection**: Alert on metric degradation

Regular evaluation ensures AI systems maintain quality, safety, and user satisfaction over time.