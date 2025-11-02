"""Data processing utilities for training."""
from typing import List, Dict, Any
import json
from pathlib import Path

from ..memory.episode_buffer import Episode


def export_episodes_for_analysis(
    episodes: List[Episode],
    output_path: str
):
    """Export episodes to JSON for analysis.
    
    Args:
        episodes: List of episodes
        output_path: Path to save JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    episodes_data = [ep.to_dict() for ep in episodes]
    
    with open(output_file, "w") as f:
        json.dump(episodes_data, f, indent=2)
    
    print(f"Exported {len(episodes)} episodes to {output_path}")


def filter_episodes_by_outcome(
    episodes: List[Episode],
    success: bool
) -> List[Episode]:
    """Filter episodes by success/failure.
    
    Args:
        episodes: List of episodes
        success: True for successful, False for failed
        
    Returns:
        Filtered list of episodes
    """
    return [
        ep for ep in episodes
        if ep.outcome.get("success", False) == success
    ]

