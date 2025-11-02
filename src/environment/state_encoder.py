"""State representation and encoding for web environments."""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class WebState:
    """Represents the state of a web environment."""
    url: str
    dom_tree: str  # Simplified DOM representation
    task_context: Dict[str, Any]
    goal: str
    available_actions: List[Dict[str, Any]]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize state to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebState":
        """Create state from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "WebState":
        """Create state from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class StateEncoder:
    """Encodes web states into consistent string representations."""
    
    def __init__(self, max_dom_length: int = 2000):
        """Initialize state encoder.
        
        Args:
            max_dom_length: Maximum length of DOM string to include
        """
        self.max_dom_length = max_dom_length
    
    def encode(self, state: WebState) -> str:
        """Encode state into a string for embedding.
        
        Returns a structured string representation suitable for embedding.
        """
        # Truncate DOM if too long
        dom = state.dom_tree
        if len(dom) > self.max_dom_length:
            dom = dom[:self.max_dom_length] + "..."
        
        # Build structured representation
        parts = [
            f"URL: {state.url}",
            f"Goal: {state.goal}",
            f"Task Context: {json.dumps(state.task_context, indent=2)}",
            f"DOM: {dom}",
            f"Available Actions: {len(state.available_actions)} actions"
        ]
        
        return "\n".join(parts)
    
    def encode_simple(self, state: WebState) -> str:
        """Encode state into a simpler string (for faster embedding)."""
        parts = [
            f"URL: {state.url}",
            f"Goal: {state.goal}",
            f"DOM summary: {self._summarize_dom(state.dom_tree)}"
        ]
        return " | ".join(parts)
    
    def _summarize_dom(self, dom: str) -> str:
        """Extract key elements from DOM."""
        # Simple heuristic: extract tag names and text content
        # In production, this could use a proper HTML parser
        if len(dom) <= 500:
            return dom
        
        # Try to extract meaningful content
        lines = dom.split('\n')[:50]  # First 50 lines
        return '\n'.join(lines)

