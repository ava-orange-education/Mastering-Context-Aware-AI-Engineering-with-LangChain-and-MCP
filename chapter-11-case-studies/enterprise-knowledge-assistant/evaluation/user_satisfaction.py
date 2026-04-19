"""
User Satisfaction Metrics

Tracks user satisfaction and engagement metrics
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class UserSatisfactionMetrics:
    """
    Tracks and analyzes user satisfaction metrics
    """
    
    def __init__(self):
        # In-memory storage for demo
        # In production, use database
        self.query_logs: List[Dict[str, Any]] = []
        self.feedback: List[Dict[str, Any]] = []
        self.click_data: List[Dict[str, Any]] = []
    
    def log_query(
        self,
        query_id: str,
        user_id: str,
        query: str,
        results_count: int,
        response_time: float
    ) -> None:
        """
        Log a search query
        
        Args:
            query_id: Query identifier
            user_id: User who made query
            query: Query text
            results_count: Number of results returned
            response_time: Time to generate response (seconds)
        """
        
        self.query_logs.append({
            "query_id": query_id,
            "user_id": user_id,
            "query": query,
            "results_count": results_count,
            "response_time": response_time,
            "timestamp": datetime.utcnow()
        })
    
    def log_feedback(
        self,
        query_id: str,
        user_id: str,
        rating: int,
        feedback_text: Optional[str] = None,
        feedback_type: str = "rating"
    ) -> None:
        """
        Log user feedback
        
        Args:
            query_id: Query identifier
            user_id: User providing feedback
            rating: Rating (1-5) or thumbs up/down (0/1)
            feedback_text: Optional text feedback
            feedback_type: Type of feedback (rating, thumbs, comment)
        """
        
        self.feedback.append({
            "query_id": query_id,
            "user_id": user_id,
            "rating": rating,
            "feedback_text": feedback_text,
            "feedback_type": feedback_type,
            "timestamp": datetime.utcnow()
        })
        
        logger.info(f"Logged feedback for query {query_id}: {rating}")
    
    def log_click(
        self,
        query_id: str,
        user_id: str,
        document_id: str,
        rank: int,
        dwell_time: Optional[float] = None
    ) -> None:
        """
        Log document click
        
        Args:
            query_id: Query identifier
            user_id: User who clicked
            document_id: Document clicked
            rank: Rank of document in results
            dwell_time: Time spent on document (seconds)
        """
        
        self.click_data.append({
            "query_id": query_id,
            "user_id": user_id,
            "document_id": document_id,
            "rank": rank,
            "dwell_time": dwell_time,
            "timestamp": datetime.utcnow()
        })
    
    def calculate_metrics(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate satisfaction metrics
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary of metrics
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Filter recent data
        recent_queries = [
            q for q in self.query_logs
            if q["timestamp"] > cutoff
        ]
        
        recent_feedback = [
            f for f in self.feedback
            if f["timestamp"] > cutoff
        ]
        
        recent_clicks = [
            c for c in self.click_data
            if c["timestamp"] > cutoff
        ]
        
        metrics = {
            "total_queries": len(recent_queries),
            "avg_response_time": self._calculate_avg_response_time(recent_queries),
            "avg_results_count": self._calculate_avg_results(recent_queries),
            "feedback_rate": self._calculate_feedback_rate(recent_queries, recent_feedback),
            "avg_rating": self._calculate_avg_rating(recent_feedback),
            "satisfaction_score": self._calculate_satisfaction_score(recent_feedback),
            "click_through_rate": self._calculate_ctr(recent_queries, recent_clicks),
            "avg_rank_clicked": self._calculate_avg_rank(recent_clicks),
            "avg_dwell_time": self._calculate_avg_dwell_time(recent_clicks),
            "return_rate": self._calculate_return_rate(recent_queries),
            "period_days": days
        }
        
        logger.info(f"Calculated metrics for {days} days")
        logger.info(f"Satisfaction score: {metrics['satisfaction_score']:.2f}")
        
        return metrics
    
    def _calculate_avg_response_time(
        self,
        queries: List[Dict[str, Any]]
    ) -> float:
        """Calculate average response time"""
        
        if not queries:
            return 0.0
        
        total_time = sum(q["response_time"] for q in queries)
        return round(total_time / len(queries), 3)
    
    def _calculate_avg_results(
        self,
        queries: List[Dict[str, Any]]
    ) -> float:
        """Calculate average number of results"""
        
        if not queries:
            return 0.0
        
        total_results = sum(q["results_count"] for q in queries)
        return round(total_results / len(queries), 1)
    
    def _calculate_feedback_rate(
        self,
        queries: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]]
    ) -> float:
        """Calculate feedback rate (% of queries with feedback)"""
        
        if not queries:
            return 0.0
        
        queries_with_feedback = len(set(f["query_id"] for f in feedback))
        return round(queries_with_feedback / len(queries), 3)
    
    def _calculate_avg_rating(
        self,
        feedback: List[Dict[str, Any]]
    ) -> float:
        """Calculate average rating"""
        
        ratings = [f["rating"] for f in feedback if f["feedback_type"] == "rating"]
        
        if not ratings:
            return 0.0
        
        return round(sum(ratings) / len(ratings), 2)
    
    def _calculate_satisfaction_score(
        self,
        feedback: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall satisfaction score (0-100)
        Based on ratings and thumbs up/down
        """
        
        if not feedback:
            return 0.0
        
        # Normalize different feedback types
        normalized_scores = []
        
        for f in feedback:
            if f["feedback_type"] == "rating":
                # 1-5 scale -> 0-100
                normalized = (f["rating"] - 1) / 4 * 100
                normalized_scores.append(normalized)
            elif f["feedback_type"] == "thumbs":
                # 0/1 -> 0/100
                normalized = f["rating"] * 100
                normalized_scores.append(normalized)
        
        if not normalized_scores:
            return 0.0
        
        return round(sum(normalized_scores) / len(normalized_scores), 1)
    
    def _calculate_ctr(
        self,
        queries: List[Dict[str, Any]],
        clicks: List[Dict[str, Any]]
    ) -> float:
        """Calculate click-through rate"""
        
        if not queries:
            return 0.0
        
        queries_with_clicks = len(set(c["query_id"] for c in clicks))
        return round(queries_with_clicks / len(queries), 3)
    
    def _calculate_avg_rank(
        self,
        clicks: List[Dict[str, Any]]
    ) -> float:
        """Calculate average rank of clicked documents"""
        
        if not clicks:
            return 0.0
        
        total_rank = sum(c["rank"] for c in clicks)
        return round(total_rank / len(clicks), 1)
    
    def _calculate_avg_dwell_time(
        self,
        clicks: List[Dict[str, Any]]
    ) -> float:
        """Calculate average dwell time on documents"""
        
        dwell_times = [c["dwell_time"] for c in clicks if c["dwell_time"] is not None]
        
        if not dwell_times:
            return 0.0
        
        return round(sum(dwell_times) / len(dwell_times), 1)
    
    def _calculate_return_rate(
        self,
        queries: List[Dict[str, Any]]
    ) -> float:
        """Calculate rate of users making multiple queries"""
        
        if not queries:
            return 0.0
        
        users = [q["user_id"] for q in queries]
        unique_users = set(users)
        
        # Users who made more than one query
        returning_users = sum(1 for user in unique_users if users.count(user) > 1)
        
        return round(returning_users / len(unique_users), 3)
    
    def get_user_metrics(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get metrics for specific user
        
        Args:
            user_id: User identifier
            days: Number of days to analyze
        
        Returns:
            User-specific metrics
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        user_queries = [
            q for q in self.query_logs
            if q["user_id"] == user_id and q["timestamp"] > cutoff
        ]
        
        user_feedback = [
            f for f in self.feedback
            if f["user_id"] == user_id and f["timestamp"] > cutoff
        ]
        
        user_clicks = [
            c for c in self.click_data
            if c["user_id"] == user_id and c["timestamp"] > cutoff
        ]
        
        return {
            "user_id": user_id,
            "total_queries": len(user_queries),
            "avg_results": self._calculate_avg_results(user_queries),
            "feedback_given": len(user_feedback),
            "avg_rating": self._calculate_avg_rating(user_feedback),
            "documents_clicked": len(user_clicks),
            "avg_dwell_time": self._calculate_avg_dwell_time(user_clicks)
        }
    
    def get_popular_queries(
        self,
        days: int = 7,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most popular queries
        
        Args:
            days: Number of days to analyze
            top_n: Number of top queries to return
        
        Returns:
            List of popular queries with counts
        """
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent_queries = [
            q["query"] for q in self.query_logs
            if q["timestamp"] > cutoff
        ]
        
        # Count occurrences
        query_counts = defaultdict(int)
        for query in recent_queries:
            query_counts[query] += 1
        
        # Sort by count
        sorted_queries = sorted(
            query_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {"query": query, "count": count}
            for query, count in sorted_queries
        ]