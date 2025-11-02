"""Episode buffer for collecting and managing agent episodes."""
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from ..environment.state_encoder import WebState


@dataclass
class Episode:
    """Represents a single episode step."""
    episode_id: str
    task_id: str
    step: int
    state: Dict[str, Any]  # Serialized WebState
    action: Dict[str, Any]
    next_state: Dict[str, Any]  # Serialized WebState
    outcome: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize episode to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create episode from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Episode":
        """Create episode from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class EpisodeBuffer:
    """Buffer for storing agent episodes."""
    
    def __init__(self, buffer_size: int = 1000, save_path: str = "data/episodes/"):
        """Initialize episode buffer.
        
        Args:
            buffer_size: Maximum number of episodes to keep in memory
            save_path: Directory path for persisting episodes
        """
        self.buffer_size = buffer_size
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.episodes: List[Episode] = []
        self.episode_counter = 0
        self.task_episode_counts: Dict[str, int] = {}  # Track episodes per task
    
    def add_episode(
        self,
        task_id: str,
        state: WebState,
        action: Dict[str, Any],
        next_state: WebState,
        outcome: Dict[str, Any]
    ) -> Episode:
        """Add a new episode to the buffer.
        
        Args:
            task_id: ID of the task
            state: Current state
            action: Action taken
            next_state: Resulting state
            outcome: Outcome of the action
            
        Returns:
            Created Episode object
        """
        # Track episode count per task
        if task_id not in self.task_episode_counts:
            self.task_episode_counts[task_id] = 0
        self.task_episode_counts[task_id] += 1
        
        episode_id = f"ep_{self.episode_counter:06d}_task_{task_id}"
        step = self.task_episode_counts[task_id]
        
        episode = Episode(
            episode_id=episode_id,
            task_id=task_id,
            step=step,
            state=state.to_dict(),
            action=action,
            next_state=next_state.to_dict(),
            outcome=outcome,
            timestamp=time.time()
        )
        
        self.episodes.append(episode)
        self.episode_counter += 1
        
        # Maintain buffer size
        if len(self.episodes) > self.buffer_size:
            # Save old episodes before removing
            self._save_episodes_batch(self.episodes[:len(self.episodes) - self.buffer_size])
            self.episodes = self.episodes[-self.buffer_size:]
        
        return episode
    
    def get_recent_episodes(self, n: int, task_id: Optional[str] = None) -> List[Episode]:
        """Get the most recent N episodes.
        
        Args:
            n: Number of episodes to retrieve
            task_id: Optional task ID to filter by
            
        Returns:
            List of recent episodes
        """
        episodes = self.episodes
        if task_id:
            episodes = [ep for ep in episodes if ep.task_id == task_id]
        
        return episodes[-n:] if len(episodes) >= n else episodes
    
    def get_episodes_for_reflection(self, n: int = 150) -> List[Episode]:
        """Get episodes for reflection phase (last N episodes).
        
        Args:
            n: Number of episodes to retrieve
            
        Returns:
            List of episodes for reflection
        """
        return self.get_recent_episodes(n)
    
    def clear(self):
        """Clear the in-memory buffer (after saving)."""
        if self.episodes:
            self._save_episodes_batch(self.episodes)
        self.episodes = []
    
    def save_all(self):
        """Save all episodes to disk."""
        self._save_episodes_batch(self.episodes)
    
    def _save_episodes_batch(self, episodes: List[Episode]):
        """Save a batch of episodes to disk."""
        if not episodes:
            return
        
        # Group by task_id for organization
        by_task: Dict[str, List[Episode]] = {}
        for ep in episodes:
            if ep.task_id not in by_task:
                by_task[ep.task_id] = []
            by_task[ep.task_id].append(ep)
        
        # Save each task's episodes
        for task_id, task_episodes in by_task.items():
            task_dir = self.save_path / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            for episode in task_episodes:
                episode_file = task_dir / f"{episode.episode_id}.json"
                with open(episode_file, "w") as f:
                    f.write(episode.to_json())
    
    def load_episodes(self, task_id: Optional[str] = None) -> List[Episode]:
        """Load episodes from disk.
        
        Args:
            task_id: Optional task ID to filter by
            
        Returns:
            List of loaded episodes
        """
        episodes = []
        
        if task_id:
            task_dir = self.save_path / task_id
            if task_dir.exists():
                for episode_file in task_dir.glob("*.json"):
                    with open(episode_file, "r") as f:
                        episodes.append(Episode.from_json(f.read()))
        else:
            # Load all episodes
            for task_dir in self.save_path.iterdir():
                if task_dir.is_dir():
                    for episode_file in task_dir.glob("*.json"):
                        with open(episode_file, "r") as f:
                            episodes.append(Episode.from_json(f.read()))
        
        return sorted(episodes, key=lambda x: x.timestamp)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored episodes.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_episodes": len(self.episodes),
            "total_saved": sum(1 for _ in self.save_path.rglob("*.json")),
            "episodes_per_task": self.task_episode_counts,
            "buffer_size": self.buffer_size
        }

