# Model I/O Module (Save/Load)
"""Model saving/loading module."""

import torch
from pathlib import Path
from typing import Any


class ModelIO:
    """Utility class for saving and loading PyTorch models with metadata."""
    
    @staticmethod
    def load_model(model, checkpoint_path: str, device: str = 'cpu') -> tuple:
        """
        Load model weights and metadata from checkpoint.
        
        Args:
            model: PyTorch model architecture
            checkpoint_path: Path to checkpoint file
            device: Device to load model to
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model, checkpoint
    
    @staticmethod
    def save_model(model, save_path: str, model_name: str = 'model', 
                   metrics: dict = None, **metadata) -> None:
        """
        Save model weights and metadata to checkpoint.
        
        Args:
            model: PyTorch model
            save_path: Path to save checkpoint
            model_name: Name of the model
            metrics: Optional dict of evaluation metrics
            **metadata: Additional metadata to save
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        checkpoint.update(metadata)
        
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")
