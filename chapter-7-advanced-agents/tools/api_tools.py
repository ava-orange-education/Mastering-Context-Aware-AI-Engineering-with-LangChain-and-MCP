"""
Tools for making API calls and web requests.
"""

from typing import Dict, Any, Optional
from .tool_base import Tool, ToolParameter
import logging
import requests

logger = logging.getLogger(__name__)


class APICallTool(Tool):
    """Tool for making REST API calls"""
    
    def __init__(self, base_url: Optional[str] = None, default_headers: Optional[Dict] = None):
        super().__init__(
            name="api_call",
            description="Make REST API calls (GET, POST, PUT, DELETE)"
        )
        
        self.parameters = [
            ToolParameter(
                name="method",
                type="string",
                description="HTTP method: GET, POST, PUT, DELETE"
            ),
            ToolParameter(
                name="url",
                type="string",
                description="API endpoint URL"
            ),
            ToolParameter(
                name="headers",
                type="object",
                description="Request headers",
                required=False
            ),
            ToolParameter(
                name="body",
                type="object",
                description="Request body (for POST/PUT)",
                required=False
            ),
            ToolParameter(
                name="params",
                type="object",
                description="Query parameters",
                required=False
            )
        ]
        
        self.base_url = base_url
        self.default_headers = default_headers or {}
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Execute API call"""
        self.validate_input(input_data)
        
        method = input_data['method'].upper()
        url = input_data['url']
        
        # Prepend base URL if configured
        if self.base_url and not url.startswith('http'):
            url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        
        headers = {**self.default_headers, **input_data.get('headers', {})}
        params = input_data.get('params', {})
        body = input_data.get('body', {})
        
        logger.info(f"Making {method} request to {url}")
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=body, params=params, timeout=30)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=body, params=params, timeout=30)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params, timeout=30)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            return {
                "status_code": response.status_code,
                "success": response.ok,
                "data": response_data,
                "headers": dict(response.headers)
            }
        
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            return {"error": "Request timeout"}
        
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {"error": str(e)}


class WebScraperTool(Tool):
    """Tool for scraping web pages"""
    
    def __init__(self):
        super().__init__(
            name="web_scraper",
            description="Extract content from web pages"
        )
        
        self.parameters = [
            ToolParameter(
                name="url",
                type="string",
                description="Web page URL"
            ),
            ToolParameter(
                name="selector",
                type="string",
                description="CSS selector to extract (optional)",
                required=False
            )
        ]
    
    def execute(self, input_data: Dict[str, Any]) -> Any:
        """Scrape web page"""
        self.validate_input(input_data)
        
        url = input_data['url']
        selector = input_data.get('selector')
        
        logger.info(f"Scraping {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if selector:
                # Extract specific elements
                elements = soup.select(selector)
                content = [elem.get_text(strip=True) for elem in elements]
            else:
                # Extract all text
                content = soup.get_text(separator='\n', strip=True)
            
            return {
                "url": url,
                "content": content,
                "title": soup.title.string if soup.title else None
            }
        
        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
            return {"error": str(e)}