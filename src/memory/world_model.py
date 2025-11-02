"""Implicit world model for predicting next states from state+action."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer


class WorldModel(nn.Module):
    """Neural network model for predicting next state embeddings from state+action."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        state_dim: int = 384,
        action_dim: int = 384,
        hidden_dim: int = 512,
        num_layers: int = 3,
        output_dim: Optional[int] = None
    ):
        """Initialize world model.
        
        Args:
            embedding_model: Name of embedding model for action encoding
            state_dim: Dimension of state embeddings
            action_dim: Dimension of action embeddings
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            output_dim: Output dimension (defaults to state_dim)
        """
        super().__init__()
        
        self.embedding_model_name = embedding_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim or state_dim
        
        # Initialize embedding model for actions
        self.action_embedder = SentenceTransformer(embedding_model)
        
        # Input: concatenated state and action embeddings
        input_dim = state_dim + action_dim
        
        # Build MLP
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def encode_action(self, action: Dict[str, Any]) -> torch.Tensor:
        """Encode action into embedding.
        
        Args:
            action: Action dictionary
            
        Returns:
            Action embedding tensor
        """
        # Convert action to text representation
        action_text = self._action_to_text(action)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.action_embedder.encode(
                action_text,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        
        # Ensure correct dimension
        if embedding.shape[0] != self.action_dim:
            if embedding.shape[0] > self.action_dim:
                embedding = embedding[:self.action_dim]
            else:
                padding = torch.zeros(self.action_dim - embedding.shape[0])
                padding = padding.to(embedding.device)
                embedding = torch.cat([embedding, padding])
        
        return embedding
    
    def forward(self, state_embedding: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict next state embedding.
        
        Args:
            state_embedding: Current state embedding
            action_embedding: Action embedding
            
        Returns:
            Predicted next state embedding
        """
        # Concatenate state and action
        combined = torch.cat([state_embedding, action_embedding], dim=-1)
        
        # Predict next state
        next_state_embedding = self.model(combined)
        
        return next_state_embedding
    
    def predict(
        self,
        state_embedding: torch.Tensor,
        action: Dict[str, Any]
    ) -> torch.Tensor:
        """Predict next state embedding for a given state and action.
        
        Args:
            state_embedding: Current state embedding
            action: Action to take
            
        Returns:
            Predicted next state embedding
        """
        self.eval()
        with torch.no_grad():
            action_embedding = self.encode_action(action)
            if state_embedding.dim() == 1:
                state_embedding = state_embedding.unsqueeze(0)
            if action_embedding.dim() == 1:
                action_embedding = action_embedding.unsqueeze(0)
            
            next_state = self.forward(state_embedding, action_embedding)
        return next_state
    
    def _action_to_text(self, action: Dict[str, Any]) -> str:
        """Convert action dictionary to text representation.
        
        Args:
            action: Action dictionary
            
        Returns:
            Action text
        """
        action_type = action.get("type", "unknown")
        parts = [f"action_type: {action_type}"]
        
        if "selector" in action:
            parts.append(f"selector: {action['selector']}")
        if "text" in action:
            parts.append(f"text: {action['text']}")
        if "value" in action:
            parts.append(f"value: {action['value']}")
        if "url" in action:
            parts.append(f"url: {action['url']}")
        
        return ", ".join(parts)
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), save_path)
        
        # Save config
        config_path = save_path.with_suffix('.config.json')
        config = {
            "embedding_model": self.embedding_model_name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "WorldModel":
        """Load model from disk.
        
        Args:
            path: Path to model file
            device: Device to load model on
            
        Returns:
            Loaded WorldModel instance
        """
        model_path = Path(path)
        config_path = model_path.with_suffix('.config.json')
        
        # Load config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                "embedding_model": "all-MiniLM-L6-v2",
                "state_dim": 384,
                "action_dim": 384,
                "hidden_dim": 512,
                "output_dim": 384
            }
        
        # Create model
        model = cls(
            embedding_model=config["embedding_model"],
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config.get("num_layers", 3),
            output_dim=config.get("output_dim", config["state_dim"])
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        return model


class WorldModelTrainer:
    """Trainer for the implicit world model."""
    
    def __init__(
        self,
        model: WorldModel,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """Initialize trainer.
        
        Args:
            model: WorldModel instance
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Embedder for states
        self.state_embedder = SentenceTransformer(model.embedding_model_name)
    
    def train_epoch(
        self,
        state_embeddings: List[np.ndarray],
        actions: List[Dict[str, Any]],
        next_state_embeddings: List[np.ndarray]
    ) -> float:
        """Train for one epoch.
        
        Args:
            state_embeddings: List of state embeddings
            actions: List of action dictionaries
            next_state_embeddings: List of next state embeddings
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Create batches
        for i in range(0, len(state_embeddings), self.batch_size):
            batch_states = state_embeddings[i:i + self.batch_size]
            batch_actions = actions[i:i + self.batch_size]
            batch_next_states = next_state_embeddings[i:i + self.batch_size]
            
            # Convert to tensors
            state_tensors = torch.tensor(np.array(batch_states), dtype=torch.float32).to(self.device)
            next_state_tensors = torch.tensor(np.array(batch_next_states), dtype=torch.float32).to(self.device)
            
            # Encode actions
            action_tensors = []
            for action in batch_actions:
                action_emb = self.model.encode_action(action)
                action_tensors.append(action_emb)
            action_tensors = torch.stack(action_tensors).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_next_states = self.model(state_tensors, action_tensors)
            
            # Compute loss
            loss = self.criterion(predicted_next_states, next_state_tensors)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

