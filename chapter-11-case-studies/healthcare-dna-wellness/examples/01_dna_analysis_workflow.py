"""
Example 1: DNA Analysis Workflow

Demonstrates complete DNA analysis workflow from variant input to interpretation
"""

import asyncio
import sys
sys.path.append('../..')

from agents.dna_analysis_agent import DNAAnalysisAgent
from shared.utils import setup_logging

setup_logging()


async def main():
    """Run DNA analysis workflow"""
    
    print("\n" + "="*70)
    print("Healthcare DNA Wellness - DNA Analysis Workflow")
    print("="*70)
    
    # Initialize agent
    print("\n1. Initializing DNA Analysis Agent...")
    agent = DNAAnalysisAgent()
    
    # Sample patient data
    patient_data = {
        "patient_id": "P123456",
        "genetic_variants": [
            {
                "gene": "MTHFR",
                "variant": "C677T",
                "zygosity": "heterozygous",
                "quality_score": 0.95
            },
            {
                "gene": "APOE",
                "variant": "E4",
                "zygosity": "heterozygous",
                "quality_score": 0.98
            },
            {
                "gene": "CYP2C19",
                "variant": "*2",
                "zygosity": "homozygous",
                "quality_score": 0.92
            }
        ]
    }
    
    print(f"\n2. Patient: {patient_data['patient_id']}")
    print(f"   Variants to analyze: {len(patient_data['genetic_variants'])}")
    
    for variant in patient_data['genetic_variants']:
        print(f"   - {variant['gene']} {variant['variant']} ({variant['zygosity']})")
    
    # Analyze variants
    print("\n3. Analyzing genetic variants...")
    result = await agent.process(patient_data)
    
    # Display results
    print("\n" + "="*70)
    print("Analysis Results")
    print("="*70)
    
    print(f"\nConfidence Score: {result.confidence:.2%}")
    print(f"Requires Review: {result.metadata.get('requires_review', False)}")
    
    print("\nVariant Analysis:")
    print("-" * 70)
    
    for i, variant in enumerate(result.metadata["variants"], 1):
        print(f"\n{i}. {variant['gene']} {variant['variant']}")
        print(f"   Zygosity: {variant['zygosity']}")
        print(f"   Quality Score: {variant.get('quality_score', 'N/A')}")
        
        if "database_info" in variant:
            db_info = variant["database_info"]
            print(f"   Clinical Significance: {db_info.get('clinical_significance', 'Unknown')}")
            print(f"   Population Frequency: {db_info.get('frequency', 'Unknown')}")
            print(f"   Associated Conditions: {', '.join(db_info.get('conditions', []))}")
        
        print(f"\n   Interpretation:")
        interpretation = variant.get('interpretation', 'No interpretation available')
        for line in interpretation.split('\n'):
            if line.strip():
                print(f"   {line.strip()}")
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(result.content)
    
    # Health check
    print("\n" + "="*70)
    print("Agent Health Check")
    print("="*70)
    
    is_healthy = await agent.health_check()
    print(f"Agent Status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
    
    print("\n" + "="*70)
    print("Workflow Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())