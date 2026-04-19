"""
Context Merger

Merges context from multiple documents intelligently
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ContextMerger:
    """
    Intelligently merges context from multiple documents
    """
    
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
    
    def merge_contexts(
        self,
        search_results: List[Any],
        query: str,
        strategy: str = "ranked"
    ) -> str:
        """
        Merge contexts from multiple search results
        
        Args:
            search_results: List of SearchResult objects
            query: Original query for relevance
            strategy: Merging strategy (ranked, interleaved, clustered)
        
        Returns:
            Merged context string
        """
        
        if not search_results:
            return ""
        
        if strategy == "ranked":
            context = self._merge_ranked(search_results)
        elif strategy == "interleaved":
            context = self._merge_interleaved(search_results)
        elif strategy == "clustered":
            context = self._merge_clustered(search_results)
        else:
            context = self._merge_ranked(search_results)
        
        # Truncate if needed
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return context
    
    def _merge_ranked(self, results: List[Any]) -> str:
        """Merge by relevance ranking"""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            doc = result.document
            
            # Format document context
            doc_context = f"""
[Document {i}] {doc.metadata.get('title', 'Untitled')}
Source: {doc.metadata.get('source', 'Unknown')}
Relevance: {result.score:.2f}

{doc.content}

---
"""
            
            # Check if adding this would exceed limit
            if current_length + len(doc_context) > self.max_context_length:
                # Add partial content
                remaining = self.max_context_length - current_length
                if remaining > 200:  # Only add if meaningful amount remains
                    partial_content = doc.content[:remaining-100] + "..."
                    doc_context = f"""
[Document {i}] {doc.metadata.get('title', 'Untitled')} (partial)
Source: {doc.metadata.get('source', 'Unknown')}

{partial_content}
"""
                    context_parts.append(doc_context)
                break
            
            context_parts.append(doc_context)
            current_length += len(doc_context)
        
        return "\n".join(context_parts)
    
    def _merge_interleaved(self, results: List[Any]) -> str:
        """Merge by interleaving content from different sources"""
        
        # Group by source
        by_source = {}
        for result in results:
            source = result.document.metadata.get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)
        
        # Interleave
        context_parts = []
        max_per_source = max(len(docs) for docs in by_source.values())
        
        for i in range(max_per_source):
            for source, docs in by_source.items():
                if i < len(docs):
                    result = docs[i]
                    doc = result.document
                    
                    context_parts.append(f"""
[{source}] {doc.metadata.get('title', 'Untitled')}
{doc.content[:500]}...
---
""")
        
        return "\n".join(context_parts)
    
    def _merge_clustered(self, results: List[Any]) -> str:
        """Merge by clustering similar content"""
        
        # Simplified clustering by topic/department
        clusters = {}
        
        for result in results:
            doc = result.document
            department = doc.metadata.get("department", "general")
            
            if department not in clusters:
                clusters[department] = []
            clusters[department].append(result)
        
        # Format by cluster
        context_parts = []
        
        for department, docs in clusters.items():
            context_parts.append(f"\n=== {department.upper()} ===\n")
            
            for result in docs[:3]:  # Top 3 per cluster
                doc = result.document
                context_parts.append(f"""
{doc.metadata.get('title', 'Untitled')}
{doc.content[:300]}...
---
""")
        
        return "\n".join(context_parts)
    
    def summarize_sources(self, results: List[Any]) -> Dict[str, Any]:
        """
        Summarize information about sources
        
        Args:
            results: Search results
        
        Returns:
            Summary statistics
        """
        
        sources = {}
        departments = {}
        total_docs = len(results)
        
        for result in results:
            doc = result.document
            
            # Count by source
            source = doc.metadata.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
            
            # Count by department
            dept = doc.metadata.get("department", "unknown")
            departments[dept] = departments.get(dept, 0) + 1
        
        return {
            "total_documents": total_docs,
            "sources": sources,
            "departments": departments,
            "avg_relevance": sum(r.score for r in results) / total_docs if total_docs > 0 else 0
        }