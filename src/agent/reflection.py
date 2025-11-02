"""Reflection system for extracting rules and anti-rules from episodes."""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..memory.episode_buffer import Episode, EpisodeBuffer
from ..memory.episodic_memory import EpisodicMemory
from ..environment.state_encoder import StateEncoder, WebState


class ReflectionSystem:
    """System for reflecting on episodes and extracting rules/anti-rules."""
    
    def __init__(
        self,
        episode_buffer: EpisodeBuffer,
        episodic_memory: EpisodicMemory,
        state_encoder: StateEncoder,
        reflection_path: str = "data/reflections/",
        llm_callback: Optional[callable] = None
    ):
        """Initialize reflection system.
        
        Args:
            episode_buffer: Buffer containing episodes
            episodic_memory: Memory system to populate
            state_encoder: Encoder for state representation
            reflection_path: Path to save reflection outputs
            llm_callback: Optional callback for LLM inference (for rule extraction)
        """
        self.episode_buffer = episode_buffer
        self.episodic_memory = episodic_memory
        self.state_encoder = state_encoder
        self.reflection_path = Path(reflection_path)
        self.reflection_path.mkdir(parents=True, exist_ok=True)
        self.llm_callback = llm_callback
    
    def reflect_on_episodes(self, episodes: Optional[List[Episode]] = None) -> Dict[str, Any]:
        """Perform reflection on a set of episodes.
        
        Args:
            episodes: Episodes to reflect on (defaults to recent episodes)
            
        Returns:
            Dictionary with reflection results
        """
        if episodes is None:
            # Get last 150 episodes for reflection
            episodes = self.episode_buffer.get_episodes_for_reflection(150)
        
        if not episodes:
            return {"error": "No episodes to reflect on"}
        
        # Group episodes by outcome
        successful_episodes = [ep for ep in episodes if ep.outcome.get("success", False)]
        failed_episodes = [ep for ep in episodes if not ep.outcome.get("success", False)]
        
        # Extract rules and anti-rules
        rules = self._extract_rules(successful_episodes)
        anti_rules = self._extract_anti_rules(failed_episodes)
        
        # Generate reflections for each state-action pair
        reflections = []
        for episode in episodes:
            state = WebState.from_dict(episode.state)
            state_text = self.state_encoder.encode(state)
            
            # Generate reflection for this state
            reflection = self._generate_reflection(
                state_text,
                episode,
                rules,
                anti_rules
            )
            
            # Add to episodic memory
            self.episodic_memory.add_memory(
                state_text=state_text,
                reflection=reflection,
                episode_id=episode.episode_id,
                task_id=episode.task_id,
                metadata={
                    "step": episode.step,
                    "outcome": episode.outcome
                }
            )
            
            reflections.append({
                "episode_id": episode.episode_id,
                "state_text": state_text,
                "reflection": reflection,
                "rules_applied": rules.get(episode.state.get("url", ""), []),
                "anti_rules_applied": anti_rules.get(episode.state.get("url", ""), [])
            })
        
        # Save reflections
        self._save_reflections(reflections)
        
        return {
            "num_episodes": len(episodes),
            "successful": len(successful_episodes),
            "failed": len(failed_episodes),
            "rules_extracted": sum(len(r) for r in rules.values()),
            "anti_rules_extracted": sum(len(r) for r in anti_rules.values()),
            "reflections_generated": len(reflections)
        }
    
    def _extract_rules(self, episodes: List[Episode]) -> Dict[str, List[str]]:
        """Extract successful patterns (rules) from episodes.
        
        Args:
            episodes: Successful episodes
            
        Returns:
            Dictionary mapping state contexts to rules
        """
        rules: Dict[str, List[str]] = {}
        
        for episode in episodes:
            state = WebState.from_dict(episode.state)
            state_url = state.url
            
            if state_url not in rules:
                rules[state_url] = []
            
            # Extract pattern: what action worked in this state?
            action_type = episode.action.get("type", "unknown")
            action_desc = self._describe_action(episode.action)
            
            rule = f"At {state_url}: {action_desc} leads to success"
            
            # Check if similar rule already exists
            if rule not in rules[state_url]:
                rules[state_url].append(rule)
        
        return rules
    
    def _extract_anti_rules(self, episodes: List[Episode]) -> Dict[str, List[str]]:
        """Extract failure patterns (anti-rules) from episodes.
        
        Args:
            episodes: Failed episodes
            
        Returns:
            Dictionary mapping state contexts to anti-rules
        """
        anti_rules: Dict[str, List[str]] = {}
        
        for episode in episodes:
            state = WebState.from_dict(episode.state)
            state_url = state.url
            
            if state_url not in anti_rules:
                anti_rules[state_url] = []
            
            # Extract pattern: what action failed in this state?
            action_type = episode.action.get("type", "unknown")
            action_desc = self._describe_action(episode.action)
            error = episode.outcome.get("error", "unknown error")
            
            anti_rule = f"At {state_url}: {action_desc} leads to failure ({error})"
            
            # Check if similar anti-rule already exists
            if anti_rule not in anti_rules[state_url]:
                anti_rules[state_url].append(anti_rule)
        
        return anti_rules
    
    def _generate_reflection(
        self,
        state_text: str,
        episode: Episode,
        rules: Dict[str, List[str]],
        anti_rules: Dict[str, List[str]]
    ) -> str:
        """Generate reflection text combining rules and anti-rules.
        
        Args:
            state_text: State representation
            episode: Episode to reflect on
            rules: Extracted rules
            anti_rules: Extracted anti-rules
            state_url: Current state URL
            
        Returns:
            Reflection text
        """
        state = WebState.from_dict(episode.state)
        state_url = state.url
        
        # Get relevant rules and anti-rules for this state
        relevant_rules = rules.get(state_url, [])
        relevant_anti_rules = anti_rules.get(state_url, [])
        
        # Build reflection text
        reflection_parts = []
        
        if relevant_rules:
            reflection_parts.append("Rules (what works):")
            for rule in relevant_rules[:3]:  # Limit to top 3
                reflection_parts.append(f"  - {rule}")
        
        if relevant_anti_rules:
            reflection_parts.append("\nAnti-rules (what doesn't work):")
            for anti_rule in relevant_anti_rules[:3]:  # Limit to top 3
                reflection_parts.append(f"  - {anti_rule}")
        
        # Add episode-specific context
        outcome = episode.outcome
        if outcome.get("success"):
            reflection_parts.append(f"\nIn this episode: Action {self._describe_action(episode.action)} succeeded.")
        else:
            error = outcome.get("error", "unknown")
            reflection_parts.append(f"\nIn this episode: Action {self._describe_action(episode.action)} failed ({error}).")
        
        return "\n".join(reflection_parts)
    
    def _describe_action(self, action: Dict[str, Any]) -> str:
        """Generate human-readable description of an action.
        
        Args:
            action: Action dictionary
            
        Returns:
            Action description string
        """
        action_type = action.get("type", "unknown")
        
        if action_type == "click":
            selector = action.get("selector", "element")
            text = action.get("text", "")
            return f"click on {selector}" + (f" ({text})" if text else "")
        
        elif action_type == "type":
            selector = action.get("selector") or action.get("name", "field")
            text = action.get("text", "")
            return f"type '{text}' into {selector}"
        
        elif action_type == "select":
            selector = action.get("selector", "dropdown")
            value = action.get("value", "")
            return f"select '{value}' from {selector}"
        
        elif action_type == "navigate":
            url = action.get("url", "")
            return f"navigate to {url}"
        
        elif action_type == "go_back":
            return "navigate back"
        
        elif action_type == "wait":
            seconds = action.get("seconds", 1)
            return f"wait {seconds}s"
        
        else:
            return f"{action_type} action"
    
    def _save_reflections(self, reflections: List[Dict[str, Any]]):
        """Save reflections to disk.
        
        Args:
            reflections: List of reflection dictionaries
        """
        if not reflections:
            return
        
        # Group by task_id
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for reflection in reflections:
            episode_id = reflection["episode_id"]
            # Extract task_id from episode_id or metadata
            task_id = "unknown"
            for ep in self.episode_buffer.episodes:
                if ep.episode_id == episode_id:
                    task_id = ep.task_id
                    break
            
            if task_id not in by_task:
                by_task[task_id] = []
            by_task[task_id].append(reflection)
        
        # Save each task's reflections
        for task_id, task_reflections in by_task.items():
            task_dir = self.reflection_path / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            reflection_file = task_dir / f"reflections_{len(task_reflections)}.json"
            with open(reflection_file, "w") as f:
                json.dump(task_reflections, f, indent=2)

