"""Base agent with episode recording and learning capabilities."""
import time
from typing import Dict, Any, Optional, List
import logging

from ..environment.web_env import WebEnvironment
from ..environment.task_definitions import Task
from ..environment.state_encoder import WebState, StateEncoder
from ..memory.episode_buffer import EpisodeBuffer
from ..memory.episodic_memory import EpisodicMemory
from ..memory.world_model import WorldModel
from .reflection import ReflectionSystem
from .action_selector import ActionSelector
from ..utils.config import Config


logger = logging.getLogger(__name__)


class BaseAgent:
    """Base agent that learns from runtime experience."""
    
    def __init__(
        self,
        config: Config,
        llm_callback: Optional[callable] = None
    ):
        """Initialize agent.
        
        Args:
            config: Configuration object
            llm_callback: Optional callback for LLM inference
        """
        self.config = config
        self.llm_callback = llm_callback
        
        # Initialize components
        self.env = WebEnvironment(
            headless=config.environment.headless,
            browser=config.environment.browser,
            viewport_width=config.environment.viewport_width,
            viewport_height=config.environment.viewport_height,
            navigation_timeout=config.environment.navigation_timeout
        )
        
        self.state_encoder = StateEncoder()
        
        self.episode_buffer = EpisodeBuffer(
            buffer_size=config.episodes.buffer_size,
            save_path=config.episodes.save_path
        )
        
        self.episodic_memory = EpisodicMemory(
            embedding_model=config.memory.embedding_model,
            index_dim=config.memory.faiss_index_dim,
            k_neighbors=config.memory.k_neighbors,
            index_path=config.memory.index_path
        )
        
        # World model (optional, can be None initially)
        self.world_model: Optional[WorldModel] = None
        
        self.reflection_system = ReflectionSystem(
            episode_buffer=self.episode_buffer,
            episodic_memory=self.episodic_memory,
            state_encoder=self.state_encoder,
            reflection_path=config.episodes.reflection_path,
            llm_callback=llm_callback
        )
        
        self.action_selector = ActionSelector(
            episodic_memory=self.episodic_memory,
            state_encoder=self.state_encoder,
            world_model=self.world_model,
            llm_callback=llm_callback,
            use_world_model=False  # Enable after training
        )
        
        # Episode tracking
        self.total_episodes = 0
        self.current_task: Optional[Task] = None
    
    def load_world_model(self, model_path: str, device: str = "cpu"):
        """Load a trained world model.
        
        Args:
            model_path: Path to saved world model
            device: Device to load model on
        """
        try:
            self.world_model = WorldModel.load(model_path, device=device)
            self.action_selector.world_model = self.world_model
            self.action_selector.use_world_model = True
            logger.info(f"Loaded world model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load world model: {e}")
    
    def run_task(self, task: Task, max_steps: int = 100) -> Dict[str, Any]:
        """Run a task and collect episodes.
        
        Args:
            task: Task to execute
            max_steps: Maximum number of steps
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Starting task: {task.task_id} - {task.goal}")
        
        self.current_task = task
        
        # Reset environment
        initial_state = self.env.reset(task)
        
        states = [initial_state]
        actions = []
        outcomes = []
        step = 0
        
        while step < max_steps:
            # Get current state
            current_state = states[-1]
            
            # Check if task is complete
            is_complete, completion_info = self.env.check_task_completion()
            if is_complete:
                logger.info(f"Task completed at step {step}")
                break
            
            # Select action
            available_actions = current_state.available_actions
            action, selection_info = self.action_selector.select_action(
                current_state,
                available_actions
            )
            
            # Execute action
            next_state, outcome = self.env.execute_action(action)
            
            # Record episode
            episode = self.episode_buffer.add_episode(
                task_id=task.task_id,
                state=current_state,
                action=action,
                next_state=next_state,
                outcome=outcome
            )
            
            self.total_episodes += 1
            states.append(next_state)
            actions.append(action)
            outcomes.append(outcome)
            
            # Check if we need to trigger sleep/reflection phase
            if self.total_episodes % self.config.agent.sleep_episode_interval == 0:
                logger.info(f"Triggering reflection phase after {self.total_episodes} episodes")
                self._sleep_and_reflect()
            
            # If action failed, log it
            if not outcome.get("success", False):
                logger.warning(f"Action failed at step {step}: {outcome.get('error', 'unknown')}")
            
            step += 1
        
        # Final check
        is_complete, completion_info = self.env.check_task_completion()
        
        result = {
            "task_id": task.task_id,
            "completed": is_complete,
            "completion_info": completion_info,
            "num_steps": step,
            "total_episodes": self.total_episodes,
            "final_state": states[-1].to_dict() if states else None
        }
        
        logger.info(f"Task {task.task_id} finished: completed={is_complete}, steps={step}")
        
        return result
    
    def _sleep_and_reflect(self):
        """Trigger sleep phase: reflect on recent episodes."""
        logger.info("Entering sleep/reflection phase...")
        
        # Get recent episodes for reflection
        episodes = self.episode_buffer.get_episodes_for_reflection(
            self.config.agent.sleep_episode_interval
        )
        
        if not episodes:
            logger.warning("No episodes to reflect on")
            return
        
        # Perform reflection
        reflection_results = self.reflection_system.reflect_on_episodes(episodes)
        
        # Save episodic memory
        self.episodic_memory.save()
        
        # Save episode buffer
        self.episode_buffer.save_all()
        
        logger.info(f"Reflection complete: {reflection_results}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_episodes": self.total_episodes,
            "episode_buffer": self.episode_buffer.get_statistics(),
            "episodic_memory": self.episodic_memory.get_statistics(),
            "world_model_loaded": self.world_model is not None
        }
    
    def shutdown(self):
        """Clean up resources."""
        logger.info("Shutting down agent...")
        
        # Save all data
        self.episode_buffer.save_all()
        self.episodic_memory.save()
        
        # Close environment
        self.env.stop()
        
        logger.info("Agent shut down complete")

