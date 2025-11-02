"""Training utilities for world model."""
import logging
from typing import List
import numpy as np
from tqdm import tqdm

from ..memory.world_model import WorldModel, WorldModelTrainer
from ..memory.episode_buffer import Episode, EpisodeBuffer
from ..environment.state_encoder import StateEncoder
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def prepare_training_data(
    episodes: List[Episode],
    state_encoder: StateEncoder,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> tuple:
    """Prepare training data from episodes.
    
    Args:
        episodes: List of episodes
        state_encoder: Encoder for states
        embedding_model: Embedding model name
        
    Returns:
        Tuple of (state_embeddings, actions, next_state_embeddings)
    """
    logger.info(f"Preparing training data from {len(episodes)} episodes")
    
    embedder = SentenceTransformer(embedding_model)
    
    state_embeddings = []
    actions = []
    next_state_embeddings = []
    
    for episode in tqdm(episodes, desc="Processing episodes"):
        try:
            # Encode state
            from ..environment.state_encoder import WebState
            state = WebState.from_dict(episode.state)
            state_text = state_encoder.encode(state)
            state_emb = embedder.encode(state_text, convert_to_numpy=True)
            
            # Encode next state
            next_state = WebState.from_dict(episode.next_state)
            next_state_text = state_encoder.encode(next_state)
            next_state_emb = embedder.encode(next_state_text, convert_to_numpy=True)
            
            state_embeddings.append(state_emb)
            actions.append(episode.action)
            next_state_embeddings.append(next_state_emb)
            
        except Exception as e:
            logger.warning(f"Failed to process episode {episode.episode_id}: {e}")
            continue
    
    logger.info(f"Prepared {len(state_embeddings)} training examples")
    
    return state_embeddings, actions, next_state_embeddings


def train_world_model(
    episode_buffer: EpisodeBuffer,
    config,
    num_epochs: int = 10,
    device: str = "cpu"
) -> WorldModel:
    """Train world model on collected episodes.
    
    Args:
        episode_buffer: Buffer containing episodes
        config: Configuration object
        num_epochs: Number of training epochs
        device: Device to train on
        
    Returns:
        Trained WorldModel
    """
    logger.info("Starting world model training")
    
    # Load all episodes
    all_episodes = episode_buffer.episodes
    if not all_episodes:
        # Try loading from disk
        all_episodes = episode_buffer.load_episodes()
    
    if not all_episodes:
        raise ValueError("No episodes available for training")
    
    logger.info(f"Training on {len(all_episodes)} episodes")
    
    # Prepare training data
    state_encoder = StateEncoder()
    state_embeddings, actions, next_state_embeddings = prepare_training_data(
        all_episodes,
        state_encoder,
        config.memory.embedding_model
    )
    
    if not state_embeddings:
        raise ValueError("No valid training data prepared")
    
    # Create model
    world_model = WorldModel(
        embedding_model=config.memory.embedding_model,
        state_dim=config.memory.faiss_index_dim,
        action_dim=config.memory.faiss_index_dim,
        hidden_dim=config.world_model.hidden_dim,
        num_layers=config.world_model.num_layers,
        output_dim=config.memory.faiss_index_dim
    ).to(device)
    
    # Create trainer
    trainer = WorldModelTrainer(
        model=world_model,
        learning_rate=config.world_model.learning_rate,
        batch_size=config.world_model.batch_size,
        device=device
    )
    
    # Train
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(state_embeddings, actions, next_state_embeddings)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    
    # Save model
    world_model.save(config.world_model.checkpoint_path)
    logger.info(f"Saved world model to {config.world_model.checkpoint_path}")
    
    return world_model

