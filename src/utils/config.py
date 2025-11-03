"""Configuration management using Pydantic."""
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Agent configuration."""
    model_name: str = "local_model"
    sleep_episode_interval: int = 150
    max_episodes_per_task: int = 1000
    action_timeout: float = 10.0


class EnvironmentConfig(BaseModel):
    """Web environment configuration."""
    headless: bool = True
    browser: str = "chromium"
    viewport_width: int = 1280
    viewport_height: int = 720
    navigation_timeout: int = 30000


class MemoryConfig(BaseModel):
    """Memory system configuration."""
    faiss_index_dim: int = 384
    k_neighbors: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    index_path: str = "data/faiss_indexes/episodic_memory.index"


class WorldModelConfig(BaseModel):
    """World model configuration."""
    hidden_dim: int = 512
    num_layers: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 32
    checkpoint_path: str = "data/checkpoints/world_model.pt"


class EpisodesConfig(BaseModel):
    """Episode storage configuration."""
    buffer_size: int = 1000
    save_path: str = "data/episodes/"
    reflection_path: str = "data/reflections/"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    log_file: str = "logs/agent.log"


class LLMConfig(BaseModel):
    """LLM model configuration."""
    model_name: str = "meta-llama/Llama-3.1-8B"
    enabled: bool = False  # Set to True to auto-initialize Llama model
    device: Optional[str] = None  # None for auto-detect
    use_quantization: bool = True  # Use 4-bit quantization (CUDA only)
    use_auth_token: Optional[str] = None  # Hugging Face token (get from env or set here)
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True


class ALFWorldConfig(BaseModel):
    """ALFWorld environment configuration."""
    enabled: bool = False  # Set to True to use ALFWorld instead of web environment
    env_type: str = "AlfredTWEnv"  # Options: AlfredTWEnv, AlfredThorEnv, AlfredHybrid
    data_dir: Optional[str] = None  # None for default ~/.cache/alfworld/
    train_eval: str = "train"  # train or eval split


class Config(BaseModel):
    """Main configuration object."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    episodes: EpisodesConfig = Field(default_factory=EpisodesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    alfworld: ALFWorldConfig = Field(default_factory=ALFWorldConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

