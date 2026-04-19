"""
Example 2: Clinical Guideline Query

Demonstrates querying clinical guidelines for genetic variants
"""

import asyncio
import sys
sys.path.append('../..')

from agents.clinical_guidelines_agent import ClinicalGuidelinesAgent
from shared.utils import setup_logging

setup_logging()


async def example_variant_query():
    """Query guidelines for specific genetic variant"""
    
    print("\n" + "="*70)
    print("Example 1: Query Guidelines for Genetic Variant")
    print("="*70)
    
    agent = ClinicalGuidelinesAgent()
    
    # Query for MTHFR variant
    query_data = {
        "variants": [
            {
                "gene": "MTHFR",
                "variant": "C677T",
                "zygosity": "heterozygous"
            }
        ],
        "conditions": ["elevated homocysteine", "folate metabolism"],
        "top_k": 5
    }
    
    print(f"\nSearching guidelines for: MTHFR C677T")
    print(f"Related conditions: {', '.join(query_data['conditions'])}")
    
    result = await agent.process(query_data)
    
    print(f"\nConfidence Score: {result.confidence:.2%}")
    print(f"Sources Found: {len(result.sources)}")
    
    print("\n" + "-"*70)
    print("Clinical Guidelines Summary")
    print("-"*70)
    print(result.content)
    
    if result.metadata.get("citations"):
        print("\n" + "-"*70)
        print("Citations")
        print("-"*70)
        
        for i, citation in enumerate(result.metadata["citations"], 1):
            print(f"\n{i}. {citation['source']} ({citation.get('year', 'N/A')})")
            print(f"   Evidence Level: {citation.get('evidence_level', 'Unknown')}")
            print(f"   Relevance Score: {citation.get('relevance_score', 0):.3f}")


async def example_condition_query():
    """Query guidelines for medical condition"""
    
    print("\n" + "="*70)
    print("Example 2: Query Guidelines for Medical Condition")
    print("="*70)
    
    agent = ClinicalGuidelinesAgent()
    
    query_data = {
        "query": "folate supplementation for MTHFR polymorphism management",
        "top_k": 5
    }
    
    print(f"\nQuery: {query_data['query']}")
    
    result = await agent.process(query_data)
    
    print(f"\nConfidence Score: {result.confidence:.2%}")
    
    print("\n" + "-"*70)
    print("Guidelines Found")
    print("-"*70)
    print(result.content)


async def example_multiple_variants():
    """Query guidelines for multiple variants"""
    
    print("\n" + "="*70)
    print("Example 3: Query Guidelines for Multiple Variants")
    print("="*70)
    
    agent = ClinicalGuidelinesAgent()
    
    query_data = {
        "variants": [
            {
                "gene": "APOE",
                "variant": "E4",
                "zygosity": "heterozygous"
            },
            {
                "gene": "MTHFR",
                "variant": "C677T",
                "zygosity": "heterozygous"
            }
        ],
        "conditions": ["cardiovascular health", "cognitive function"],
        "top_k": 8
    }
    
    print("\nSearching guidelines for multiple variants:")
    for variant in query_data["variants"]:
        print(f"  - {variant['gene']} {variant['variant']}")
    
    result = await agent.process(query_data)
    
    print(f"\nConfidence Score: {result.confidence:.2%}")
    print(f"Guidelines Retrieved: {len(result.sources)}")
    
    print("\n" + "-"*70)
    print("Integrated Guidelines Summary")
    print("-"*70)
    print(result.content)


async def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("Clinical Guideline Query Examples")
    print("="*70)
    
    await example_variant_query()
    
    await example_condition_query()
    
    await example_multiple_variants()
    
    print("\n" + "="*70)
    print("All Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())