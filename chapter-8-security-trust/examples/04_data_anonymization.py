"""
Example 4: Data Anonymization
Demonstrates PII detection and various anonymization strategies.
"""

import sys
sys.path.append('..')

from data_protection.pii_detector import PIIDetector
from data_protection.data_anonymizer import DataAnonymizer


def main():
    print("=" * 70)
    print("  DATA ANONYMIZATION EXAMPLE")
    print("=" * 70 + "\n")
    
    # Initialize components
    pii_detector = PIIDetector()
    anonymizer = DataAnonymizer(pii_detector)
    
    # Test document with various PII types
    test_document = """
    Customer Information:
    Name: John Smith
    Email: john.smith@company.com
    Phone: 555-123-4567
    SSN: 123-45-6789
    Address: 123 Main Street, Springfield
    Credit Card: 4532-1234-5678-9010
    
    Notes: Customer called regarding account issue. Provided callback number 555-987-6543.
    """
    
    print("Phase 1: PII Detection")
    print("-" * 70)
    print(f"Original document:\n{test_document}\n")
    
    # Detect PII
    pii_scan = pii_detector.scan_document(test_document)
    
    print(f"PII detected: {pii_scan['contains_pii']}")
    print(f"Total PII instances: {pii_scan['pii_count']}")
    print(f"\nPII types found:")
    for pii_type, count in pii_scan['summary'].items():
        print(f"  - {pii_type}: {count}")
    
    print(f"\nDetailed matches:")
    for match in pii_scan['matches']:
        print(f"  - {match['type']}: {match['value']} (confidence: {match['confidence']})")
    
    print("\n\nPhase 2: Anonymization Strategies")
    print("-" * 70)
    
    strategies = ['redact', 'mask', 'pseudonymize', 'hash']
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.upper()}")
        print("-" * 40)
        
        result = anonymizer.anonymize(test_document, strategy=strategy)
        
        print(f"Anonymized text:\n{result['anonymized_text'][:300]}...\n")
        print(f"Replacements made: {result['replacements']}")
        print(f"PII detected: {result['pii_detected']}")
    
    print("\n\nPhase 3: Reversible Pseudonymization")
    print("-" * 70)
    
    # Pseudonymize
    anonymizer.clear_mapping()  # Clear previous mappings
    result = anonymizer.anonymize(test_document, strategy='pseudonymize')
    
    print(f"Pseudonymized:\n{result['anonymized_text'][:200]}...\n")
    
    # Demonstrate consistency
    result2 = anonymizer.anonymize(test_document, strategy='pseudonymize')
    print("Consistency check (same input should get same pseudonyms):")
    print(f"Match: {result['anonymized_text'] == result2['anonymized_text']}")
    
    # Deanonymize (only works with pseudonymization)
    deanonymized = anonymizer.deanonymize(result['anonymized_text'])
    if deanonymized:
        print(f"\nDeanonymized matches original: {deanonymized == test_document}")
    
    print("\n\nPhase 4: Selective Anonymization")
    print("-" * 70)
    
    # Detect high-risk PII
    has_high_risk = pii_detector.has_high_risk_pii(test_document)
    print(f"Contains high-risk PII (SSN, credit cards): {has_high_risk}")
    
    if has_high_risk:
        print("\nApplying redaction for high-risk PII...")
        result = anonymizer.anonymize(test_document, strategy='redact')
        print(f"Result:\n{result['anonymized_text'][:200]}...")
    
    print("\n" + "=" * 70)
    print("  DATA ANONYMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()