"""
Example 3: Wellness Plan Generation

Demonstrates complete wellness plan generation with safety checks
"""

import asyncio
import sys
sys.path.append('../..')

from agents.dna_analysis_agent import DNAAnalysisAgent
from agents.clinical_guidelines_agent import ClinicalGuidelinesAgent
from agents.wellness_recommendation_agent import WellnessRecommendationAgent
from agents.compliance_audit_agent import ComplianceAuditAgent
from evaluation.medical_accuracy_validator import MedicalAccuracyValidator
from evaluation.safety_checker import SafetyChecker
from shared.utils import setup_logging

setup_logging()


async def generate_wellness_plan():
    """Generate complete wellness plan"""
    
    print("\n" + "="*70)
    print("Healthcare DNA Wellness - Complete Wellness Plan Generation")
    print("="*70)
    
    # Initialize agents
    print("\n1. Initializing agents...")
    dna_agent = DNAAnalysisAgent()
    clinical_agent = ClinicalGuidelinesAgent()
    wellness_agent = WellnessRecommendationAgent()
    compliance_agent = ComplianceAuditAgent()
    accuracy_validator = MedicalAccuracyValidator()
    safety_checker = SafetyChecker()
    
    print("   ✅ All agents initialized")
    
    # Patient data
    patient_data = {
        "patient_id": "P789012",
        "genetic_variants": [
            {
                "gene": "MTHFR",
                "variant": "C677T",
                "zygosity": "homozygous",
                "quality_score": 0.96
            },
            {
                "gene": "APOE",
                "variant": "E4",
                "zygosity": "heterozygous",
                "quality_score": 0.94
            }
        ]
    }
    
    patient_context = {
        "age": 45,
        "conditions": ["hypertension"],
        "medications": ["lisinopril"],
        "allergies": []
    }
    
    print(f"\n2. Patient Information:")
    print(f"   ID: {patient_data['patient_id']}")
    print(f"   Age: {patient_context['age']}")
    print(f"   Conditions: {', '.join(patient_context['conditions'])}")
    print(f"   Medications: {', '.join(patient_context['medications'])}")
    print(f"   Variants: {len(patient_data['genetic_variants'])}")
    
    # Step 1: Analyze DNA
    print("\n3. Analyzing genetic variants...")
    dna_result = await dna_agent.process(patient_data)
    print(f"   ✅ DNA analysis complete (confidence: {dna_result.confidence:.2%})")
    
    # Step 2: Retrieve clinical guidelines
    print("\n4. Retrieving clinical guidelines...")
    guidelines_input = {
        "variants": dna_result.metadata["variants"],
        "conditions": patient_context["conditions"],
        "top_k": 5
    }
    
    guidelines_result = await clinical_agent.process(guidelines_input)
    print(f"   ✅ Retrieved {len(guidelines_result.sources)} guidelines")
    
    # Step 3: Generate recommendations
    print("\n5. Generating wellness recommendations...")
    wellness_input = {
        "analysis": dna_result,
        "guidelines": guidelines_result,
        "patient_context": patient_context,
        "focus_areas": ["nutrition", "supplements", "lifestyle", "monitoring"]
    }
    
    wellness_result = await wellness_agent.process(wellness_input)
    print(f"   ✅ Recommendations generated (confidence: {wellness_result.confidence:.2%})")
    
    # Step 4: Validate medical accuracy
    print("\n6. Validating medical accuracy...")
    accuracy_validation = await accuracy_validator.validate_recommendations(
        wellness_result.content,
        patient_context
    )
    
    print(f"   Accuracy Score: {accuracy_validation['accuracy_score']:.2%}")
    print(f"   Valid: {accuracy_validation['overall_valid']}")
    
    if accuracy_validation["issues"]:
        print(f"   ⚠️  Issues found: {len(accuracy_validation['issues'])}")
        for issue in accuracy_validation["issues"]:
            print(f"      - {issue.get('message', 'Unknown issue')}")
    
    if accuracy_validation["warnings"]:
        print(f"   ⚠️  Warnings: {len(accuracy_validation['warnings'])}")
        for warning in accuracy_validation["warnings"]:
            print(f"      - {warning}")
    
    # Step 5: Safety check
    print("\n7. Performing safety check...")
    safety_check = await safety_checker.check_safety(
        wellness_result.content,
        patient_context
    )
    
    print(f"   Safety Score: {safety_check['safety_score']:.2%}")
    print(f"   Safe: {safety_check['safe']}")
    print(f"   Requires Physician Review: {safety_check['requires_physician_review']}")
    
    if safety_check["concerns"]:
        print(f"   ⚠️  Safety concerns: {len(safety_check['concerns'])}")
        for concern in safety_check["concerns"]:
            print(f"      - [{concern.get('severity', 'unknown')}] {concern.get('message', 'Unknown concern')}")
    
    # Step 6: Compliance audit
    print("\n8. Logging compliance audit...")
    compliance_result = await compliance_agent.process({
        "action": "generate_wellness_plan",
        "user_id": "doctor_smith",
        "patient_id": patient_data["patient_id"],
        "data_accessed": {
            "genetic_data": True,
            "clinical_data": True
        },
        "output_data": wellness_result.content,
        "ip_address": "192.168.1.100"
    })
    
    print(f"   ✅ Audit logged")
    
    if not compliance_result.metadata["compliant"]:
        print("   ⚠️  Compliance issues detected!")
    
    # Display results
    print("\n" + "="*70)
    print("WELLNESS PLAN")
    print("="*70)
    
    print("\nGenetic Analysis Summary:")
    print("-" * 70)
    print(dna_result.content)
    
    print("\n\nClinical Guidelines Summary:")
    print("-" * 70)
    print(guidelines_result.content)
    
    print("\n\nPersonalized Recommendations:")
    print("-" * 70)
    print(wellness_result.content)
    
    # Final summary
    print("\n" + "="*70)
    print("QUALITY ASSURANCE SUMMARY")
    print("="*70)
    
    print(f"\n✓ Medical Accuracy: {accuracy_validation['accuracy_score']:.2%}")
    print(f"✓ Safety Score: {safety_check['safety_score']:.2%}")
    print(f"✓ Confidence Score: {wellness_result.confidence:.2%}")
    print(f"✓ HIPAA Compliant: {compliance_result.metadata['compliant']}")
    
    requires_review = (
        not safety_check["safe"] or
        not accuracy_validation["overall_valid"] or
        safety_check["requires_physician_review"]
    )
    
    print(f"\n{'⚠️  REQUIRES PHYSICIAN REVIEW' if requires_review else '✅ READY FOR PATIENT'}")
    
    print("\n" + "="*70)
    print("Wellness Plan Generation Complete!")
    print("="*70)


async def main():
    """Run example"""
    await generate_wellness_plan()


if __name__ == "__main__":
    asyncio.run(main())