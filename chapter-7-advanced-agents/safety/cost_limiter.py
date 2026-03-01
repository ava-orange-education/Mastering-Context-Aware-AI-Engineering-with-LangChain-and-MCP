"""
Cost limiting and budget management for agent operations.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CostLimiter:
    """Manages cost budgets for agent operations"""
    
    def __init__(self, daily_budget: float = 100.0, monthly_budget: float = 1000.0):
        """
        Initialize cost limiter
        
        Args:
            daily_budget: Daily budget in dollars
            monthly_budget: Monthly budget in dollars
        """
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        
        self.daily_costs: Dict[str, float] = {}  # date -> cost
        self.monthly_costs: Dict[str, float] = {}  # month -> cost
        
        self.total_costs = 0.0
        self.cost_breakdown: Dict[str, float] = {}  # operation_type -> cost
    
    def record_cost(self, operation_type: str, cost: float):
        """
        Record operation cost
        
        Args:
            operation_type: Type of operation (e.g., 'llm_call', 'tool_execution')
            cost: Cost in dollars
        """
        today = datetime.now().strftime('%Y-%m-%d')
        this_month = datetime.now().strftime('%Y-%m')
        
        # Update daily costs
        if today not in self.daily_costs:
            self.daily_costs[today] = 0.0
        self.daily_costs[today] += cost
        
        # Update monthly costs
        if this_month not in self.monthly_costs:
            self.monthly_costs[this_month] = 0.0
        self.monthly_costs[this_month] += cost
        
        # Update total
        self.total_costs += cost
        
        # Update breakdown
        if operation_type not in self.cost_breakdown:
            self.cost_breakdown[operation_type] = 0.0
        self.cost_breakdown[operation_type] += cost
        
        logger.debug(f"Recorded cost: ${cost:.4f} for {operation_type}")
    
    def check_budget(self) -> Dict[str, Any]:
        """
        Check if within budget limits
        
        Returns:
            Budget status
        """
        today = datetime.now().strftime('%Y-%m-%d')
        this_month = datetime.now().strftime('%Y-%m')
        
        daily_spent = self.daily_costs.get(today, 0.0)
        monthly_spent = self.monthly_costs.get(this_month, 0.0)
        
        daily_remaining = self.daily_budget - daily_spent
        monthly_remaining = self.monthly_budget - monthly_spent
        
        within_budget = daily_remaining > 0 and monthly_remaining > 0
        
        return {
            'within_budget': within_budget,
            'daily': {
                'budget': self.daily_budget,
                'spent': daily_spent,
                'remaining': daily_remaining,
                'percentage': (daily_spent / self.daily_budget * 100) if self.daily_budget > 0 else 0
            },
            'monthly': {
                'budget': self.monthly_budget,
                'spent': monthly_spent,
                'remaining': monthly_remaining,
                'percentage': (monthly_spent / self.monthly_budget * 100) if self.monthly_budget > 0 else 0
            }
        }
    
    def can_afford(self, estimated_cost: float) -> bool:
        """
        Check if can afford estimated cost
        
        Args:
            estimated_cost: Estimated operation cost
            
        Returns:
            Whether operation is affordable
        """
        budget_status = self.check_budget()
        
        daily_remaining = budget_status['daily']['remaining']
        monthly_remaining = budget_status['monthly']['remaining']
        
        return (daily_remaining >= estimated_cost and 
                monthly_remaining >= estimated_cost)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        budget_status = self.check_budget()
        
        return {
            'total_costs': self.total_costs,
            'budget_status': budget_status,
            'cost_breakdown': self.cost_breakdown,
            'top_expenses': self._get_top_expenses(5)
        }
    
    def _get_top_expenses(self, n: int) -> List[Dict[str, float]]:
        """Get top N expense categories"""
        sorted_expenses = sorted(
            self.cost_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'operation': op, 'cost': cost}
            for op, cost in sorted_expenses[:n]
        ]
    
    def reset_daily_budget(self):
        """Reset daily budget (call this daily)"""
        today = datetime.now().strftime('%Y-%m-%d')
        if today in self.daily_costs:
            del self.daily_costs[today]
        logger.info("Reset daily budget")


class RateLimiter:
    """Rate limiting for agent operations"""
    
    def __init__(self, max_requests_per_minute: int = 60, 
                 max_requests_per_hour: int = 1000):
        """
        Initialize rate limiter
        
        Args:
            max_requests_per_minute: Max requests per minute
            max_requests_per_hour: Max requests per hour
        """
        self.max_per_minute = max_requests_per_minute
        self.max_per_hour = max_requests_per_hour
        
        self.request_timestamps: List[datetime] = []
    
    def check_rate_limit(self) -> Dict[str, Any]:
        """
        Check if within rate limits
        
        Returns:
            Rate limit status
        """
        now = datetime.now()
        
        # Remove old timestamps
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if ts > hour_ago
        ]
        
        # Count recent requests
        requests_last_minute = sum(
            1 for ts in self.request_timestamps
            if ts > minute_ago
        )
        
        requests_last_hour = len(self.request_timestamps)
        
        # Check limits
        within_minute_limit = requests_last_minute < self.max_per_minute
        within_hour_limit = requests_last_hour < self.max_per_hour
        
        return {
            'allowed': within_minute_limit and within_hour_limit,
            'per_minute': {
                'limit': self.max_per_minute,
                'used': requests_last_minute,
                'remaining': self.max_per_minute - requests_last_minute
            },
            'per_hour': {
                'limit': self.max_per_hour,
                'used': requests_last_hour,
                'remaining': self.max_per_hour - requests_last_hour
            }
        }
    
    def record_request(self):
        """Record a request"""
        self.request_timestamps.append(datetime.now())
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        import time
        
        while True:
            status = self.check_rate_limit()
            
            if status['allowed']:
                break
            
            # Wait 1 second and retry
            logger.info("Rate limit reached, waiting...")
            time.sleep(1)