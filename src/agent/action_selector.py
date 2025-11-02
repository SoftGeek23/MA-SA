"""Action selector using FAISS kNN retrieval and world model predictions."""
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np

from ..memory.episodic_memory import EpisodicMemory
from ..memory.world_model import WorldModel
from ..environment.state_encoder import WebState, StateEncoder
from sentence_transformers import SentenceTransformer


class ActionSelector:
    """Selects actions using episodic memory retrieval and optionally world model."""
    
    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        state_encoder: StateEncoder,
        world_model: Optional[WorldModel] = None,
        llm_callback: Optional[callable] = None,
        use_world_model: bool = False
    ):
        """Initialize action selector.
        
        Args:
            episodic_memory: Episodic memory for retrieval
            state_encoder: Encoder for state representation
            world_model: Optional world model for next-state prediction
            llm_callback: Callback for LLM-based action generation
            use_world_model: Whether to use world model for action selection
        """
        self.episodic_memory = episodic_memory
        self.state_encoder = state_encoder
        self.world_model = world_model
        self.llm_callback = llm_callback
        self.use_world_model = use_world_model and world_model is not None
        
        # Embedder for states (used for world model)
        if self.use_world_model:
            self.state_embedder = SentenceTransformer(
                world_model.embedding_model_name
            )
    
    def select_action(
        self,
        current_state: WebState,
        available_actions: List[Dict[str, Any]],
        k: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select an action given the current state.
        
        Args:
            current_state: Current state of the environment
            available_actions: List of available actions
            k: Number of neighbors to retrieve (defaults to memory k_neighbors)
            
        Returns:
            Tuple of (selected_action, selection_info)
        """
        # Encode current state
        state_text = self.state_encoder.encode(current_state)
        
        # Retrieve similar states with reflections
        retrieved = self.episodic_memory.search(state_text, k=k)
        
        selection_info = {
            "retrieved_memories": len(retrieved),
            "method": "fallback"
        }
        
        if retrieved:
            # Extract reflections from retrieved memories
            reflections = [entry["reflection"] for _, entry in retrieved]
            
            # Combine reflections into context
            reflection_context = "\n\n".join([
                f"Memory {i+1}:\n{reflection}"
                for i, reflection in enumerate(reflections[:3])  # Top 3
            ])
            
            selection_info["method"] = "memory_guided"
            selection_info["reflections"] = reflections[:3]
        else:
            reflection_context = "No similar past experiences found."
            selection_info["method"] = "no_memory"
        
        # If world model is available, use it to predict outcomes
        action_scores = {}
        if self.use_world_model and available_actions:
            # Encode current state
            state_embedding = self.state_embedder.encode(
                state_text, convert_to_tensor=True
            )
            
            # Predict next states for each available action
            device = next(self.world_model.parameters()).device
            state_embedding = state_embedding.to(device)
            
            for action in available_actions:
                try:
                    predicted_next_state = self.world_model.predict(
                        state_embedding, action
                    )
                    # Use some heuristic to score (e.g., distance to goal)
                    # For now, use a simple scoring
                    action_scores[action] = float(torch.norm(predicted_next_state))
                except Exception as e:
                    # If prediction fails, use default score
                    action_scores[action] = 0.0
        
        # If we have LLM callback, use it to generate action
        if self.llm_callback:
            selected_action = self._generate_action_with_llm(
                current_state,
                available_actions,
                reflection_context,
                action_scores
            )
            selection_info["generation_method"] = "llm"
        else:
            # Fallback: select based on available actions and context
            selected_action = self._select_fallback_action(
                available_actions,
                reflection_context,
                action_scores
            )
            selection_info["generation_method"] = "fallback"
        
        selection_info["selected_action"] = selected_action
        return selected_action, selection_info
    
    def _generate_action_with_llm(
        self,
        current_state: WebState,
        available_actions: List[Dict[str, Any]],
        reflection_context: str,
        action_scores: Dict[Dict[str, Any], float]
    ) -> Dict[str, Any]:
        """Generate action using LLM.
        
        Args:
            current_state: Current state
            available_actions: Available actions
            reflection_context: Context from retrieved memories
            action_scores: Scores from world model (if available)
            
        Returns:
            Selected action dictionary
        """
        # Build prompt
        prompt = self._build_action_prompt(
            current_state,
            available_actions,
            reflection_context,
            action_scores
        )
        
        # Call LLM
        if self.llm_callback:
            response = self.llm_callback(prompt)
            # Parse response into action
            action = self._parse_llm_response(response, available_actions)
            return action
        
        # Fallback if LLM fails
        return self._select_fallback_action(available_actions, reflection_context, action_scores)
    
    def _build_action_prompt(
        self,
        current_state: WebState,
        available_actions: List[Dict[str, Any]],
        reflection_context: str,
        action_scores: Dict[Dict[str, Any], float]
    ) -> str:
        """Build prompt for LLM action generation.
        
        Args:
            current_state: Current state
            available_actions: Available actions
            reflection_context: Reflection context
            action_scores: World model scores
            
        Returns:
            Prompt string
        """
        state_summary = self.state_encoder.encode_simple(current_state)
        
        actions_str = "\n".join([
            f"  {i+1}. {self._format_action(action)}"
            for i, action in enumerate(available_actions[:10])  # Limit to 10
        ])
        
        prompt = f"""You are an autonomous agent performing web tasks.

Current State:
{state_summary}

Goal: {current_state.goal}

Past Experience (what worked/didn't work):
{reflection_context}

Available Actions:
{actions_str}

Based on the current state, goal, and past experience, select the best action to take.
Respond with the action number or a JSON description of the action.

Action:"""
        
        return prompt
    
    def _format_action(self, action: Dict[str, Any]) -> str:
        """Format action for display.
        
        Args:
            action: Action dictionary
            
        Returns:
            Formatted action string
        """
        action_type = action.get("type", "unknown")
        
        if action_type == "click":
            selector = action.get("selector", "")
            text = action.get("text", "")
            return f"Click: {selector}" + (f" ({text})" if text else "")
        
        elif action_type == "type":
            selector = action.get("selector") or action.get("name", "")
            text = action.get("text", "")
            return f"Type '{text}' into: {selector}"
        
        elif action_type == "select":
            selector = action.get("selector", "")
            value = action.get("value", "")
            return f"Select '{value}' from: {selector}"
        
        elif action_type == "navigate":
            url = action.get("url", "")
            return f"Navigate to: {url}"
        
        elif action_type == "go_back":
            return "Navigate back"
        
        else:
            return f"{action_type}: {action}"
    
    def _parse_llm_response(
        self,
        response: str,
        available_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse LLM response into action dictionary.
        
        Args:
            response: LLM response string
            available_actions: Available actions
            
        Returns:
            Selected action dictionary
        """
        # Try to extract action number
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            action_idx = int(numbers[0]) - 1
            if 0 <= action_idx < len(available_actions):
                return available_actions[action_idx]
        
        # Try to parse JSON
        import json
        try:
            # Look for JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                action_dict = json.loads(json_match.group())
                # Validate and return
                if "type" in action_dict:
                    return action_dict
        except:
            pass
        
        # Fallback: return first available action
        if available_actions:
            return available_actions[0]
        
        # Last resort: return a wait action
        return {"type": "wait", "seconds": 1}
    
    def _select_fallback_action(
        self,
        available_actions: List[Dict[str, Any]],
        reflection_context: str,
        action_scores: Dict[Dict[str, Any], float]
    ) -> Dict[str, Any]:
        """Select action using fallback heuristics.
        
        Args:
            available_actions: Available actions
            reflection_context: Reflection context (for future use)
            action_scores: World model scores
            
        Returns:
            Selected action dictionary
        """
        if not available_actions:
            return {"type": "wait", "seconds": 1}
        
        # If we have world model scores, use them
        if action_scores:
            # Select action with lowest score (closest to predicted goal state)
            best_action = min(action_scores.items(), key=lambda x: x[1])[0]
            return best_action
        
        # Otherwise, use simple heuristics
        # Prefer click actions over others
        click_actions = [a for a in available_actions if a.get("type") == "click"]
        if click_actions:
            return click_actions[0]
        
        # Return first available action
        return available_actions[0]

