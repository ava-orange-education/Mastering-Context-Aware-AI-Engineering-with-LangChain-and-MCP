"""
Example 3: Cross-Department Query

Demonstrates querying across departments and finding related information
"""

import asyncio
import sys
sys.path.append('../..')

from agents.document_search_agent import DocumentSearchAgent
from agents.cross_reference_agent import CrossReferenceAgent
from agents.expert_finder_agent import ExpertFinderAgent
from access_control.organizational_hierarchy import OrganizationalHierarchy
from shared.utils import setup_logging

setup_logging()


async def cross_department_search():
    """Search across multiple departments"""
    
    print("\n" + "="*70)
    print("Example 1: Cross-Department Search")
    print("="*70)
    
    # Setup organizational hierarchy
    org = OrganizationalHierarchy()
    org.add_department("dept_eng", "Engineering")
    org.add_department("dept_prod", "Product")
    org.add_department("dept_sales", "Sales")
    
    # Search agent
    search_agent = DocumentSearchAgent()
    
    user_id = "vp@company.com"
    query = "product launch strategy"
    
    print(f"\nQuery: {query}")
    print(f"User: {user_id} (VP - cross-department access)")
    
    # Search across departments
    print("\nSearching across departments...")
    result = await search_agent.process({
        "query": query,
        "user_id": user_id,
        "user_groups": ["engineering", "product", "sales", "leadership"],
        "top_k": 15
    })
    
    print("\n" + "-"*70)
    print("Cross-Department Results")
    print("-"*70)
    
    # Simulate grouping by department
    departments_covered = ["Engineering", "Product", "Sales"]
    print(f"\nDocuments found from {len(departments_covered)} departments:")
    for dept in departments_covered:
        print(f"  • {dept}")
    
    print("\n" + "-"*70)
    print("Synthesized Answer")
    print("-"*70)
    print(result.content)


async def find_related_across_projects():
    """Find related documents across projects"""
    
    print("\n" + "="*70)
    print("Example 2: Cross-Project Document Discovery")
    print("="*70)
    
    cross_ref_agent = CrossReferenceAgent()
    
    document_id = "doc_project_alpha_001"
    user_id = "pm@company.com"
    
    print(f"\nPrimary Document: Project Alpha Planning")
    print(f"User: {user_id} (Project Manager)")
    
    print("\nFinding related documents...")
    result = await cross_ref_agent.process({
        "document_id": document_id,
        "user_id": user_id,
        "relationship_types": ["topical", "temporal", "organizational"],
        "max_results": 10
    })
    
    print("\n" + "-"*70)
    print("Related Documents Found")
    print("-"*70)
    
    if result.metadata.get("relationships"):
        relationships = result.metadata["relationships"]
        
        # Group by type
        from collections import defaultdict
        by_type = defaultdict(list)
        
        for rel in relationships:
            by_type[rel["relationship_type"]].append(rel)
        
        for rel_type, docs in by_type.items():
            print(f"\n{rel_type.capitalize()} Relationships ({len(docs)}):")
            for doc in docs[:3]:
                print(f"  • {doc['title']}")
                print(f"    {doc['explanation']}")
    
    print("\n" + "-"*70)
    print("Analysis")
    print("-"*70)
    print(result.content)


async def find_experts_cross_functional():
    """Find experts across different functions"""
    
    print("\n" + "="*70)
    print("Example 3: Cross-Functional Expert Discovery")
    print("="*70)
    
    expert_agent = ExpertFinderAgent()
    
    topic = "machine learning deployment"
    
    print(f"\nTopic: {topic}")
    print(f"Searching across all departments...")
    
    result = await expert_agent.process({
        "topic": topic,
        "department": None,  # All departments
        "min_contributions": 2,
        "max_experts": 10
    })
    
    print("\n" + "-"*70)
    print("Experts Found")
    print("-"*70)
    
    if result.metadata.get("experts"):
        experts = result.metadata["experts"]
        
        # Group by department
        from collections import defaultdict
        by_dept = defaultdict(list)
        
        for expert in experts:
            dept = expert.get("department", "Unknown")
            by_dept[dept].append(expert)
        
        for dept, dept_experts in by_dept.items():
            print(f"\n{dept} ({len(dept_experts)} experts):")
            for expert in dept_experts[:2]:
                print(f"  • {expert['author']}")
                if expert.get("email"):
                    print(f"    Email: {expert['email']}")
                print(f"    Level: {expert['expertise_level'].replace('_', ' ').title()}")
                print(f"    Contributions: {len(expert['contributions'])}")
    
    print("\n" + "-"*70)
    print("Expert Recommendations")
    print("-"*70)
    print(result.content)


async def collaboration_discovery():
    """Discover collaboration opportunities"""
    
    print("\n" + "="*70)
    print("Example 4: Collaboration Discovery")
    print("="*70)
    
    search_agent = DocumentSearchAgent()
    cross_ref_agent = CrossReferenceAgent()
    expert_agent = ExpertFinderAgent()
    
    query = "customer retention strategies"
    user_id = "strategy@company.com"
    
    print(f"\nQuery: {query}")
    print(f"Goal: Find collaboration opportunities across teams")
    
    # Step 1: Search for relevant documents
    print("\n1. Searching for relevant documents...")
    search_result = await search_agent.process({
        "query": query,
        "user_id": user_id,
        "user_groups": ["strategy", "leadership"],
        "top_k": 10
    })
    print(f"   Found {len(search_result.sources)} relevant documents")
    
    # Step 2: Find related work
    print("\n2. Finding related initiatives...")
    related_result = await cross_ref_agent.process({
        "query": query,
        "user_id": user_id,
        "relationship_types": ["organizational", "topical"],
        "max_results": 8
    })
    
    relationships = related_result.metadata.get("relationships", [])
    
    # Identify departments involved
    departments = set()
    for rel in relationships:
        dept = rel.get("source", "").split("/")[0] if "/" in rel.get("source", "") else "Unknown"
        departments.add(dept)
    
    print(f"   Found work from {len(departments)} departments")
    
    # Step 3: Find relevant experts
    print("\n3. Identifying subject matter experts...")
    expert_result = await expert_agent.process({
        "topic": query,
        "min_contributions": 2,
        "max_experts": 5
    })
    
    experts = expert_result.metadata.get("experts", [])
    print(f"   Found {len(experts)} experts")
    
    # Summary
    print("\n" + "="*70)
    print("Collaboration Opportunities")
    print("="*70)
    
    print(f"\nDepartments with relevant work:")
    for dept in sorted(departments):
        print(f"  • {dept}")
    
    print(f"\nKey experts to involve:")
    for expert in experts[:3]:
        print(f"  • {expert['author']} ({expert.get('department', 'Unknown')})")
    
    print("\n" + "-"*70)
    print("Strategic Insights")
    print("-"*70)
    print(search_result.content)
    
    print("\n" + "-"*70)
    print("Recommended Actions")
    print("-"*70)
    print("1. Organize cross-functional meeting with identified experts")
    print("2. Review related documents from all departments")
    print("3. Identify gaps and collaboration opportunities")
    print("4. Create unified strategy document")


async def main():
    """Run all cross-department examples"""
    
    print("\n" + "="*70)
    print("Cross-Department Query Examples")
    print("="*70)
    
    await cross_department_search()
    
    await find_related_across_projects()
    
    await find_experts_cross_functional()
    
    await collaboration_discovery()
    
    print("\n" + "="*70)
    print("All Cross-Department Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())