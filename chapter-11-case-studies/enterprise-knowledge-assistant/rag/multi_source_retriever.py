"""
Multi-Source Retriever

Retrieves and merges results from multiple sources
"""

from typing import List, Dict, Any, Optional
import logging
import asyncio
import sys
sys.path.append('../..')

from shared.base_rag import SearchResult
from .permission_aware_retriever import PermissionAwareRetriever

logger = logging.getLogger(__name__)


class MultiSourceRetriever:
    """
    Retriever that searches across multiple sources
    """
    
    def __init__(self):
        self.retriever = PermissionAwareRetriever()
        
        # Source weights for ranking
        self.source_weights = {
            "sharepoint": 1.0,
            "confluence": 1.0,
            "slack": 0.8,
            "google_drive": 1.0,
            "local": 0.7
        }
    
    async def initialize(self) -> None:
        """Initialize retriever"""
        await self.retriever.initialize()
    
    async def search_all_sources(
        self,
        query: str,
        user_id: str,
        user_groups: Optional[List[str]] = None,
        top_k: int = 10,
        source_weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """
        Search across all sources and merge results
        
        Args:
            query: Search query
            user_id: User performing search
            user_groups: User's groups
            top_k: Total number of results to return
            source_weights: Custom source weights
        
        Returns:
            Merged and ranked results
        """
        
        weights = source_weights or self.source_weights
        
        # Search all sources in parallel
        search_tasks = []
        
        for source in weights.keys():
            task = self.retriever.search(
                query=query,
                user_id=user_id,
                user_groups=user_groups,
                top_k=top_k,
                sources=[source]
            )
            search_tasks.append((source, task))
        
        # Gather results
        all_results = []
        
        for source, task in search_tasks:
            try:
                results = await task
                
                # Apply source weight to scores
                weight = weights.get(source, 1.0)
                for result in results:
                    result.score = result.score * weight
                    result.document.metadata["source"] = source
                
                all_results.extend(results)
            
            except Exception as e:
                logger.error(f"Failed to search {source}: {e}")
        
        # Merge and rank results
        merged_results = self._merge_results(all_results, top_k)
        
        logger.info(
            f"Retrieved {len(merged_results)} results from "
            f"{len(weights)} sources"
        )
        
        return merged_results
    
    async def search_related(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        sources: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search for related documents
        
        Args:
            query: Query or document content
            user_id: User ID
            top_k: Number of results
            sources: Sources to search
        
        Returns:
            Related documents
        """
        
        return await self.retriever.search(
            query=query,
            user_id=user_id,
            top_k=top_k,
            sources=sources
        )
    
    def _merge_results(
        self,
        all_results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """
        Merge and deduplicate results from multiple sources
        
        Args:
            all_results: Results from all sources
            top_k: Number of results to return
        
        Returns:
            Merged results
        """
        
        # Deduplicate by document ID
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            doc_id = result.document.id
            
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(result)
        
        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top k
        return unique_results[:top_k]
    
    async def search_by_department(
        self,
        query: str,
        department: str,
        user_id: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search within a specific department
        
        Args:
            query: Search query
            department: Department name
            user_id: User ID
            top_k: Number of results
        
        Returns:
            Department-specific results
        """
        
        filters = {
            "department": department
        }
        
        return await self.retriever.search(
            query=query,
            user_id=user_id,
            top_k=top_k,
            filters=filters
        )
    
    async def search_by_project(
        self,
        query: str,
        project: str,
        user_id: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Search within a specific project
        
        Args:
            query: Search query
            project: Project name
            user_id: User ID
            top_k: Number of results
        
        Returns:
            Project-specific results
        """
        
        filters = {
            "project": project
        }
        
        return await self.retriever.search(
            query=query,
            user_id=user_id,
            top_k=top_k,
            filters=filters
        )