"""Example usage of MemAgent."""
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.agent.base_agent import BaseAgent
from src.environment.task_definitions import TaskDefinition

# Example LLM callback (replace with your actual LLM)
def llm_callback(prompt: str) -> str:
    """Example LLM callback.
    
    In production, replace this with your actual 14B model inference.
    For example:
    - Using transformers library for local model
    - Using vLLM for faster inference
    - Using API calls to remote models
    """
    # Dummy implementation - replace with actual model
    print("LLM Prompt:", prompt[:200] + "...")
    return "1"  # Return action number or JSON


def main():
    """Example usage."""
    # Load config
    config = Config.from_yaml("configs/default_config.yaml")
    
    # Setup logging
    logger = setup_logger(level=config.logging.level)
    
    # Create agent
    agent = BaseAgent(config, llm_callback=llm_callback)
    
    # Create a task
    task_def = TaskDefinition()
    task = task_def.create_search_click_task(
        task_id="example_1",
        description="Example search task",
        url="https://example.com",
        search_query="example query",
        target_element="button#search"
    )
    
    try:
        # Run task
        result = agent.run_task(task, max_steps=50)
        logger.info(f"Task completed: {result}")
        
        # Print statistics
        stats = agent.get_statistics()
        logger.info(f"Agent stats: {stats}")
        
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()

