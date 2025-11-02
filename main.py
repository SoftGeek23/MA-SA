"""Main entry point for MemAgent."""
import sys
import argparse
import logging
from pathlib import Path

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.agent.base_agent import BaseAgent
from src.environment.task_definitions import TaskDefinition, TaskType
from src.training.world_model_trainer import train_world_model


def create_example_tasks() -> list:
    """Create example tasks for testing."""
    task_def = TaskDefinition()
    
    tasks = [
        # Example search+click task
        task_def.create_search_click_task(
            task_id="search_click_1",
            description="Search and click example",
            url="https://example.com",
            search_query="test query",
            target_element="button#search"
        ),
        
        # Example form-fill task
        task_def.create_form_fill_task(
            task_id="form_fill_1",
            description="Form filling example",
            url="https://example.com/form",
            form_fields={"name": "Test User", "email": "test@example.com"},
            submit_button="button[type='submit']"
        ),
    ]
    
    return tasks


def dummy_llm_callback(prompt: str) -> str:
    """Dummy LLM callback for testing.
    
    In production, this would call your actual LLM.
    """
    # For now, return a simple response
    # This should be replaced with actual LLM inference
    return "1"  # Select first action


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MemAgent - Agent Learning from Runtime Experience")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "train_world_model"],
        default="run",
        help="Mode: run agent or train world model"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task ID to run (optional)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for world model training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Setup logging
    logger = setup_logger(
        level=config.logging.level,
        log_file=config.logging.log_file
    )
    
    logger.info("Starting MemAgent")
    logger.info(f"Mode: {args.mode}")
    
    if args.mode == "train_world_model":
        # Training mode
        from src.memory.episode_buffer import EpisodeBuffer
        
        logger.info("Training world model")
        
        episode_buffer = EpisodeBuffer(
            buffer_size=config.episodes.buffer_size,
            save_path=config.episodes.save_path
        )
        
        # Load existing episodes
        episodes = episode_buffer.load_episodes()
        logger.info(f"Loaded {len(episodes)} episodes for training")
        
        if not episodes:
            logger.error("No episodes found. Run the agent first to collect episodes.")
            sys.exit(1)
        
        # Train world model
        world_model = train_world_model(
            episode_buffer,
            config,
            num_epochs=args.epochs
        )
        
        logger.info("World model training complete")
        
    else:
        # Run mode
        agent = BaseAgent(config, llm_callback=dummy_llm_callback)
        
        try:
            # Create example tasks
            tasks = create_example_tasks()
            
            # Run tasks
            for task in tasks:
                if args.task and task.task_id != args.task:
                    continue
                
                result = agent.run_task(task, max_steps=50)
                logger.info(f"Task result: {result}")
            
            # Print statistics
            stats = agent.get_statistics()
            logger.info(f"Agent statistics: {stats}")
            
        finally:
            agent.shutdown()


if __name__ == "__main__":
    main()

