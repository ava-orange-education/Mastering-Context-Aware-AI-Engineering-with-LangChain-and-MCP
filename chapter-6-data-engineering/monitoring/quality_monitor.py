"""
Data quality monitoring and validation.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataQualityRule:
    """Base class for data quality rules"""
    
    def __init__(self, rule_name: str, severity: str = "warning"):
        """
        Initialize quality rule
        
        Args:
            rule_name: Name of the rule
            severity: 'info', 'warning', or 'error'
        """
        self.rule_name = rule_name
        self.severity = severity
    
    def validate(self, data: Any) -> Dict[str, Any]:
        """
        Validate data against rule
        
        Args:
            data: Data to validate
            
        Returns:
            Validation result
        """
        raise NotImplementedError


class CompletenessRule(DataQualityRule):
    """Check for null/missing values"""
    
    def __init__(self, required_fields: List[str], threshold: float = 0.95):
        super().__init__("completeness", "warning")
        self.required_fields = required_fields
        self.threshold = threshold
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if required fields are present and non-null"""
        missing_fields = []
        
        for field in self.required_fields:
            if field not in data or data[field] is None or data[field] == "":
                missing_fields.append(field)
        
        completeness_score = 1 - (len(missing_fields) / len(self.required_fields))
        
        return {
            'rule': self.rule_name,
            'passed': completeness_score >= self.threshold,
            'score': completeness_score,
            'threshold': self.threshold,
            'missing_fields': missing_fields,
            'severity': self.severity
        }


class UniquenessRule(DataQualityRule):
    """Check for duplicate records"""
    
    def __init__(self, key_fields: List[str]):
        super().__init__("uniqueness", "error")
        self.key_fields = key_fields
        self.seen_keys = set()
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if record is unique based on key fields"""
        # Generate key from specified fields
        key_values = tuple(data.get(field) for field in self.key_fields)
        
        is_duplicate = key_values in self.seen_keys
        self.seen_keys.add(key_values)
        
        return {
            'rule': self.rule_name,
            'passed': not is_duplicate,
            'is_duplicate': is_duplicate,
            'key_fields': self.key_fields,
            'severity': self.severity
        }
    
    def reset(self):
        """Reset seen keys"""
        self.seen_keys.clear()


class ConsistencyRule(DataQualityRule):
    """Check data format consistency"""
    
    def __init__(self, field: str, pattern: str, data_type: Optional[type] = None):
        super().__init__("consistency", "warning")
        self.field = field
        self.pattern = pattern
        self.data_type = data_type
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if field matches expected pattern/type"""
        import re
        
        value = data.get(self.field)
        
        issues = []
        
        # Check data type
        if self.data_type and not isinstance(value, self.data_type):
            issues.append(f"Expected type {self.data_type.__name__}, got {type(value).__name__}")
        
        # Check pattern
        if self.pattern and isinstance(value, str):
            if not re.match(self.pattern, value):
                issues.append(f"Does not match pattern: {self.pattern}")
        
        return {
            'rule': self.rule_name,
            'passed': len(issues) == 0,
            'field': self.field,
            'issues': issues,
            'severity': self.severity
        }


class AccuracyRule(DataQualityRule):
    """Check data accuracy against reference"""
    
    def __init__(self, field: str, valid_values: Optional[List] = None, 
                 min_value: Optional[float] = None, max_value: Optional[float] = None):
        super().__init__("accuracy", "warning")
        self.field = field
        self.valid_values = valid_values
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if value is within expected range/set"""
        value = data.get(self.field)
        
        issues = []
        
        # Check against valid values
        if self.valid_values and value not in self.valid_values:
            issues.append(f"Value not in valid set: {self.valid_values}")
        
        # Check range
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                issues.append(f"Below minimum: {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                issues.append(f"Above maximum: {self.max_value}")
        
        return {
            'rule': self.rule_name,
            'passed': len(issues) == 0,
            'field': self.field,
            'value': value,
            'issues': issues,
            'severity': self.severity
        }


class TimelinessRule(DataQualityRule):
    """Check data freshness"""
    
    def __init__(self, timestamp_field: str, max_age_hours: float = 24):
        super().__init__("timeliness", "warning")
        self.timestamp_field = timestamp_field
        self.max_age_hours = max_age_hours
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if data is recent enough"""
        from datetime import datetime, timedelta
        
        timestamp = data.get(self.timestamp_field)
        
        if timestamp is None:
            return {
                'rule': self.rule_name,
                'passed': False,
                'error': f"Missing timestamp field: {self.timestamp_field}",
                'severity': self.severity
            }
        
        # Convert to datetime if string
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        age = datetime.now() - timestamp
        age_hours = age.total_seconds() / 3600
        
        return {
            'rule': self.rule_name,
            'passed': age_hours <= self.max_age_hours,
            'age_hours': age_hours,
            'max_age_hours': self.max_age_hours,
            'severity': self.severity
        }


class QualityMonitor:
    """Monitor data quality across pipeline"""
    
    def __init__(self):
        self.rules: List[DataQualityRule] = []
        self.validation_history: List[Dict[str, Any]] = []
    
    def add_rule(self, rule: DataQualityRule):
        """Add quality rule"""
        self.rules.append(rule)
        logger.info(f"Added quality rule: {rule.rule_name}")
    
    def validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate single record against all rules
        
        Args:
            record: Record to validate
            
        Returns:
            Validation results
        """
        results = {
            'record_id': record.get('record_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'rule_results': [],
            'passed': True,
            'warnings': 0,
            'errors': 0
        }
        
        for rule in self.rules:
            rule_result = rule.validate(record)
            results['rule_results'].append(rule_result)
            
            if not rule_result['passed']:
                results['passed'] = False
                
                if rule.severity == 'warning':
                    results['warnings'] += 1
                elif rule.severity == 'error':
                    results['errors'] += 1
        
        self.validation_history.append(results)
        
        return results
    
    def validate_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate batch of records
        
        Args:
            records: List of records to validate
            
        Returns:
            Batch validation summary
        """
        batch_results = {
            'total_records': len(records),
            'passed_records': 0,
            'failed_records': 0,
            'total_warnings': 0,
            'total_errors': 0,
            'validation_time': datetime.now().isoformat()
        }
        
        for record in records:
            result = self.validate_record(record)
            
            if result['passed']:
                batch_results['passed_records'] += 1
            else:
                batch_results['failed_records'] += 1
            
            batch_results['total_warnings'] += result['warnings']
            batch_results['total_errors'] += result['errors']
        
        # Calculate pass rate
        batch_results['pass_rate'] = (
            batch_results['passed_records'] / batch_results['total_records']
            if batch_results['total_records'] > 0 else 0
        )
        
        logger.info(f"Batch validation: {batch_results['pass_rate']:.2%} pass rate")
        
        return batch_results
    
    def get_quality_report(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate quality report
        
        Args:
            last_n: Optional limit to last N validations
            
        Returns:
            Quality report
        """
        history = self.validation_history[-last_n:] if last_n else self.validation_history
        
        if not history:
            return {'message': 'No validation history'}
        
        # Aggregate statistics
        total_validations = len(history)
        passed_validations = sum(1 for h in history if h['passed'])
        total_warnings = sum(h['warnings'] for h in history)
        total_errors = sum(h['errors'] for h in history)
        
        # Rule-specific statistics
        rule_stats = {}
        for validation in history:
            for rule_result in validation['rule_results']:
                rule_name = rule_result['rule']
                
                if rule_name not in rule_stats:
                    rule_stats[rule_name] = {
                        'total': 0,
                        'passed': 0,
                        'failed': 0
                    }
                
                rule_stats[rule_name]['total'] += 1
                if rule_result['passed']:
                    rule_stats[rule_name]['passed'] += 1
                else:
                    rule_stats[rule_name]['failed'] += 1
        
        # Calculate pass rates
        for rule_name in rule_stats:
            rule_stats[rule_name]['pass_rate'] = (
                rule_stats[rule_name]['passed'] / rule_stats[rule_name]['total']
            )
        
        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': total_validations - passed_validations,
            'overall_pass_rate': passed_validations / total_validations,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'rule_statistics': rule_stats,
            'report_generated': datetime.now().isoformat()
        }