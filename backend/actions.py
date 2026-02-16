from datetime import datetime
from typing import Dict, Any
import json
from ddgs import DDGS

class ActionHandler:
    """Handles tool/action execution for the agent"""
    
    def __init__(self):
        self.actions = {
            "web_search": self.web_search,
            "calculator": self.calculator,
            "get_time": self.get_time,
        }
    
    def execute(self, action_name: str, parameters: Dict[str, Any]) -> str:
        """Execute an action by name"""
        if action_name not in self.actions:
            return f"Error: Unknown action '{action_name}'"
        
        try:
            return self.actions[action_name](**parameters)
        except Exception as e:
            return f"Error executing {action_name}: {str(e)}"
    
    def web_search(self, query: str, max_results: int = 3) -> str:
        """Search the web using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
            if not results:
                return "No results found."
            
            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(f"{i}. {result['title']}\n{result['body']}\nSource: {result['href']}\n")
            
            return "\n".join(formatted)
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def calculator(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            # Only allow safe operations
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def get_time(self) -> str:
        """Get current date and time"""
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    def get_available_actions(self) -> list:
        """Return list of available actions"""
        return [
            {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {"query": "string", "max_results": "int (optional)"}
            },
            {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "parameters": {"expression": "string"}
            },
            {
                "name": "get_time",
                "description": "Get current date and time",
                "parameters": {}
            }
        ]