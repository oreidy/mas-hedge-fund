"""
Improved LSTM model for individual ticker return prediction.

This model eliminates look-ahead bias and focuses on predicting individual ticker returns
rather than portfolio-level actions or returns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path

from utils.logger import logger


class TickerLSTMModel(nn.Module):
    """
    LSTM model for individual ticker return prediction.
    
    Key improvements:
    1. Predicts individual ticker returns (regression)
    2. Handles variable number of tickers per prediction
    3. Includes ticker embedding for better generalization
    4. No look-ahead bias in architecture
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        ticker_embedding_dim: int = 16,
        device: str = None
    ):
        """
        Initialize the ticker LSTM model.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            ticker_embedding_dim: Dimension of ticker embeddings
            device: Device to run model on (cuda/cpu/mps)
        """
        super(TickerLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.ticker_embedding_dim = ticker_embedding_dim
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # LSTM layers for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Batch normalization for LSTM output
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Ticker vocabulary will be built dynamically
        self.ticker_vocab = {}
        self.ticker_embedding = None
        
        # Return prediction head
        self.return_head = nn.Sequential(
            nn.Linear(hidden_size + ticker_embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Single return prediction
        )
        
        # Move model to device
        self.to(self.device)
        
        logger.info(f"TickerLSTMModel initialized with {sum(p.numel() for p in self.parameters())} parameters on {self.device}")
    
    def build_ticker_vocabulary(self, tickers: Set[str]):
        """
        Build ticker vocabulary and embedding layer.
        
        Args:
            tickers: Set of all ticker symbols that will be used
        """
        self.ticker_vocab = {ticker: idx for idx, ticker in enumerate(sorted(tickers))}
        vocab_size = len(self.ticker_vocab)
        
        # Create ticker embedding layer
        self.ticker_embedding = nn.Embedding(vocab_size, self.ticker_embedding_dim)
        
        # Reinitialize return head to account for ticker embeddings
        self.return_head = nn.Sequential(
            nn.Linear(self.hidden_size + self.ticker_embedding_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Move new layers to device
        self.ticker_embedding.to(self.device)
        self.return_head.to(self.device)
        
        logger.info(f"Built ticker vocabulary with {vocab_size} tickers")
    
    def encode_ticker(self, ticker: str) -> int:
        """Encode ticker symbol to integer index."""
        if ticker not in self.ticker_vocab:
            raise ValueError(f"Ticker {ticker} not in vocabulary. Available tickers: {list(self.ticker_vocab.keys())}")
        return self.ticker_vocab[ticker]
    
    def forward(self, x: torch.Tensor, tickers: List[str] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
            tickers: List of ticker symbols for each sample in batch
            
        Returns:
            Predicted returns [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Apply batch normalization
        if batch_size > 1:
            last_hidden = self.batch_norm(last_hidden)
        
        # Ticker embeddings
        if tickers is not None and self.ticker_embedding is not None:
            ticker_indices = torch.tensor([self.encode_ticker(ticker) for ticker in tickers], 
                                        dtype=torch.long, device=self.device)
            ticker_embeds = self.ticker_embedding(ticker_indices)  # [batch_size, embedding_dim]
            
            # Concatenate LSTM output with ticker embeddings
            combined_features = torch.cat([last_hidden, ticker_embeds], dim=1)
        else:
            # If no ticker info, pad with zeros
            ticker_embeds = torch.zeros(batch_size, self.ticker_embedding_dim, device=self.device)
            combined_features = torch.cat([last_hidden, ticker_embeds], dim=1)
        
        # Return prediction
        predictions = self.return_head(combined_features)
        
        return predictions
    
    def predict_returns(self, x: torch.Tensor, tickers: List[str]) -> Dict[str, float]:
        """
        Predict returns for multiple tickers.
        
        Args:
            x: Input sequences [num_tickers, sequence_length, input_size]
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to predicted return
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x, tickers)
            predictions = predictions.cpu().numpy().flatten()
            
            return {ticker: float(pred) for ticker, pred in zip(tickers, predictions)}
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'ticker_embedding_dim': self.ticker_embedding_dim,
            'vocab_size': len(self.ticker_vocab) if self.ticker_vocab else 0,
            'device': str(self.device)
        }
    
    def save(self, filepath: str, include_optimizer: bool = False, optimizer: torch.optim.Optimizer = None):
        """
        Save model state to disk.
        
        Args:
            filepath: Path to save model
            include_optimizer: Whether to save optimizer state
            optimizer: Optimizer to save (if include_optimizer=True)
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'ticker_embedding_dim': self.ticker_embedding_dim
            },
            'ticker_vocab': self.ticker_vocab
        }
        
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = None) -> 'TickerLSTMModel':
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create model instance
        config = checkpoint['model_config']
        model = cls(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            ticker_embedding_dim=config['ticker_embedding_dim'],
            device=device
        )
        
        # Restore ticker vocabulary and build embedding
        ticker_vocab = checkpoint['ticker_vocab']
        if ticker_vocab:
            model.build_ticker_vocabulary(set(ticker_vocab.keys()))
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def freeze_lstm_layers(self):
        """Freeze LSTM layers for transfer learning."""
        for param in self.lstm.parameters():
            param.requires_grad = False
        logger.info("LSTM layers frozen")
    
    def unfreeze_all_layers(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen")


class TickerLSTMEnsemble(nn.Module):
    """
    Ensemble of ticker LSTM models for improved predictions.
    """
    
    def __init__(self, models: List[TickerLSTMModel], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained TickerLSTMModel instances
            weights: Optional weights for ensemble averaging
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        # Ensure all models are on the same device
        device = models[0].device
        for model in models:
            model.to(device)
        
        self.device = device
    
    def forward(self, x: torch.Tensor, tickers: List[str]) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            tickers: List of ticker symbols
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model_pred = model.forward(x, tickers)
            predictions.append(model_pred * weight)
        
        # Weighted average
        ensemble_pred = torch.stack(predictions, dim=0).sum(dim=0)
        
        return ensemble_pred
    
    def predict_returns(self, x: torch.Tensor, tickers: List[str]) -> Dict[str, float]:
        """Predict returns using ensemble."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x, tickers)
            predictions = predictions.cpu().numpy().flatten()
            
            return {ticker: float(pred) for ticker, pred in zip(tickers, predictions)}