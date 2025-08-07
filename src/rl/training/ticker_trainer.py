"""
Improved LSTM trainer for ticker return prediction with proper temporal validation.

This trainer eliminates look-ahead bias through:
1. Temporal train/validation splits (no future data in training)
2. Walk-forward validation methodology
3. Individual ticker return prediction
4. Proper early stopping based on temporal validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from datetime import datetime, timedelta

from ..models.ticker_lstm_model import TickerLSTMModel
from ..preprocessing.ticker_preprocessor import TickerPreprocessor
from ..data_collection.episode import TrainingEpisode
from utils.logger import logger


class TickerLSTMTrainer:
    """
    Trainer for ticker LSTM models with proper temporal validation.
    
    Key features:
    1. Temporal train/validation splits to prevent look-ahead bias
    2. Walk-forward validation for realistic performance assessment
    3. Early stopping based on temporal validation performance
    4. Comprehensive metrics tracking and visualization
    """
    
    def __init__(
        self,
        model: TickerLSTMModel,
        preprocessor: TickerPreprocessor,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        max_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-6,
        validation_split: float = 0.2,
        device: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: TickerLSTMModel instance
            preprocessor: Fitted TickerPreprocessor instance
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience (epochs without improvement)
            min_delta: Minimum change to qualify as improvement
            validation_split: Fraction of data for temporal validation
            device: Device to use for training
        """
        self.model = model
        self.preprocessor = preprocessor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.validation_split = validation_split
        
        # Set device
        if device is None:
            self.device = model.device
        else:
            self.device = torch.device(device)
            self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_r2': [],
            'val_r2': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        logger.info(f"TickerLSTMTrainer initialized with lr={learning_rate}, batch_size={batch_size}")
    
    def create_temporal_splits(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        metadata: List[Dict[str, Any]]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, List[str]], Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Create temporal train/validation splits to prevent look-ahead bias.
        
        Args:
            X: Input sequences [num_samples, sequence_length, num_features]
            y: Target returns [num_samples]
            metadata: Metadata for each sample containing dates and tickers
            
        Returns:
            (train_data, val_data) where each is (X, y, tickers)
        """
        # Extract dates and sort chronologically
        dates = [datetime.strptime(meta['end_date'], '%Y-%m-%d') for meta in metadata]
        sorted_indices = np.argsort(dates)
        
        # Calculate split point based on temporal ordering
        split_idx = int(len(sorted_indices) * (1 - self.validation_split))
        
        train_indices = sorted_indices[:split_idx]
        val_indices = sorted_indices[split_idx:]
        
        # Create splits
        X_train = X[train_indices]
        y_train = y[train_indices]
        tickers_train = [metadata[i]['ticker'] for i in train_indices]
        
        X_val = X[val_indices]
        y_val = y[val_indices]
        tickers_val = [metadata[i]['ticker'] for i in val_indices]
        
        # Log split information
        train_start = min(dates[i] for i in train_indices).strftime('%Y-%m-%d')
        train_end = max(dates[i] for i in train_indices).strftime('%Y-%m-%d')
        val_start = min(dates[i] for i in val_indices).strftime('%Y-%m-%d')
        val_end = max(dates[i] for i in val_indices).strftime('%Y-%m-%d')
        
        logger.info(f"Temporal split - Train: {train_start} to {train_end} ({len(train_indices)} samples)")
        logger.info(f"Temporal split - Val: {val_start} to {val_end} ({len(val_indices)} samples)")
        
        return (X_train, y_train, tickers_train), (X_val, y_val, tickers_val)
    
    def create_data_loaders(
        self, 
        train_data: Tuple[np.ndarray, np.ndarray, List[str]], 
        val_data: Tuple[np.ndarray, np.ndarray, List[str]]
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders for training and validation.
        
        Args:
            train_data: Training data (X, y, tickers)
            val_data: Validation data (X, y, tickers)
            
        Returns:
            (train_loader, val_loader)
        """
        X_train, y_train, _ = train_data
        X_val, y_val, _ = val_data
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,  # Shuffle within temporal constraints
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, tickers_train: List[str]) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            tickers_train: List of tickers for training samples (in order)
            
        Returns:
            (average_loss, mae, r2_score)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Get corresponding tickers for this batch
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(tickers_train))
            batch_tickers = tickers_train[start_idx:end_idx]
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x, batch_tickers).squeeze()
            
            # Calculate loss
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(batch_y.detach().cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        return avg_loss, mae, r2
    
    def validate_epoch(self, val_loader: DataLoader, tickers_val: List[str]) -> Tuple[float, float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            tickers_val: List of tickers for validation samples (in order)
            
        Returns:
            (average_loss, mae, r2_score)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Get corresponding tickers for this batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(tickers_val))
                batch_tickers = tickers_val[start_idx:end_idx]
                
                # Forward pass
                predictions = self.model(batch_x, batch_tickers).squeeze()
                
                # Calculate loss
                loss = self.criterion(predictions, batch_y)
                
                # Track metrics
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        return avg_loss, mae, r2
    
    def train(self, episodes: List[TrainingEpisode]) -> Dict[str, Any]:
        """
        Train the model on episodes with temporal validation.
        
        Args:
            episodes: List of training episodes (must be sorted by date)
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting ticker LSTM training with temporal validation")
        
        # Transform episodes to training data
        X, y, metadata = self.preprocessor.transform(episodes)
        
        if len(X) == 0:
            raise ValueError("No training data generated - check episodes and preprocessor settings")
        
        # Build ticker vocabulary for the model
        all_tickers = set(meta['ticker'] for meta in metadata)
        self.model.build_ticker_vocabulary(all_tickers)
        
        # Create temporal splits
        train_data, val_data = self.create_temporal_splits(X, y, metadata)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_data, val_data)
        
        # Training loop
        logger.info(f"Training for up to {self.max_epochs} epochs")
        
        for epoch in range(self.max_epochs):
            # Train epoch
            train_loss, train_mae, train_r2 = self.train_epoch(train_loader, train_data[2])
            
            # Validate epoch
            val_loss, val_mae, val_r2 = self.validate_epoch(val_loader, val_data[2])
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_r2'].append(train_r2)
            self.history['val_r2'].append(val_r2)
            self.history['learning_rates'].append(current_lr)
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
                
                logger.info(f"Epoch {epoch+1}/{self.max_epochs} - "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f} (*), "
                           f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, LR: {current_lr:.2e}")
            else:
                self.epochs_without_improvement += 1
                
                logger.info(f"Epoch {epoch+1}/{self.max_epochs} - "
                           f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                           f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, LR: {current_lr:.2e}")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with validation loss: {self.best_val_loss:.6f}")
        
        # Calculate final metrics
        final_metrics = self.evaluate(val_data)
        
        training_results = {
            'epochs_trained': len(self.history['train_loss']),
            'best_val_loss': self.best_val_loss,
            'final_metrics': final_metrics,
            'history': self.history.copy(),
            'model_info': self.model.get_model_info()
        }
        
        logger.info("Training completed successfully")
        return training_results
    
    def evaluate(self, data: Tuple[np.ndarray, np.ndarray, List[str]]) -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            data: Data tuple (X, y, tickers)
            
        Returns:
            Dictionary of evaluation metrics
        """
        X, y, tickers = data
        
        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Evaluate
        val_loss, val_mae, val_r2 = self.validate_epoch(loader, tickers)
        
        return {
            'mse': val_loss,
            'rmse': np.sqrt(val_loss),
            'mae': val_mae,
            'r2': val_r2
        }
    
    def walk_forward_validate(
        self, 
        episodes: List[TrainingEpisode], 
        initial_train_size: int = 250,
        retrain_frequency: int = 21,  # Monthly retraining (21 trading days)
        min_val_size: int = 21
    ) -> Dict[str, Any]:
        """
        Walk-forward validation with expanding window and periodic retraining.
        
        Args:
            episodes: List of all episodes (sorted by date)
            initial_train_size: Initial training set size
            retrain_frequency: How often to retrain (in episodes)
            min_val_size: Minimum validation set size
            
        Returns:
            Walk-forward validation results
        """
        if len(episodes) < initial_train_size + min_val_size:
            raise ValueError(f"Not enough episodes: need at least {initial_train_size + min_val_size}, got {len(episodes)}")
        
        logger.info(f"Starting walk-forward validation with initial_train_size={initial_train_size}")
        
        results = {
            'predictions': [],
            'actuals': [],
            'dates': [],
            'tickers': [],
            'retrain_dates': [],
            'validation_metrics': []
        }
        
        current_pos = initial_train_size
        last_retrain = 0
        
        while current_pos < len(episodes):
            # Determine if we need to retrain
            should_retrain = (current_pos - last_retrain) >= retrain_frequency
            
            if should_retrain or current_pos == initial_train_size:
                # Retrain on expanding window
                train_episodes = episodes[:current_pos]
                
                logger.info(f"Retraining at episode {current_pos} on {len(train_episodes)} episodes")
                
                # Create new preprocessor and model for this window
                temp_preprocessor = TickerPreprocessor(
                    sequence_length=self.preprocessor.sequence_length,
                    prediction_horizon=self.preprocessor.prediction_horizon,
                    min_consensus_strength=self.preprocessor.min_consensus_strength,
                    min_episodes_per_ticker=self.preprocessor.min_episodes_per_ticker
                )
                
                # Fit and transform training data
                X_train, y_train, metadata_train = temp_preprocessor.fit_transform(train_episodes)
                
                if len(X_train) == 0:
                    logger.warning(f"No training data at position {current_pos}, skipping")
                    current_pos += 1
                    continue
                
                # Create and train new model
                temp_model = TickerLSTMModel(
                    input_size=X_train.shape[2],
                    hidden_size=self.model.hidden_size,
                    num_layers=self.model.num_layers,
                    dropout=self.model.dropout,
                    device=self.device
                )
                
                # Build vocabulary and train
                all_tickers = set(meta['ticker'] for meta in metadata_train)
                temp_model.build_ticker_vocabulary(all_tickers)
                
                # Quick training with reduced epochs for walk-forward
                temp_trainer = TickerLSTMTrainer(
                    model=temp_model,
                    preprocessor=temp_preprocessor,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    max_epochs=min(20, self.max_epochs // 2),  # Reduced epochs for efficiency
                    patience=5,
                    validation_split=0.15  # Smaller validation split
                )
                
                temp_trainer.train(train_episodes)
                
                # Update current model and preprocessor
                self.model = temp_model
                self.preprocessor = temp_preprocessor
                
                last_retrain = current_pos
                results['retrain_dates'].append(episodes[current_pos].date)
            
            # Predict on next batch of episodes
            end_pos = min(current_pos + min_val_size, len(episodes))
            val_episodes = episodes[current_pos:end_pos]
            
            # Get predictions for validation episodes
            for i, episode in enumerate(val_episodes):
                if current_pos + i + 1 < len(episodes):  # Need next episode for actual returns
                    next_episode = episodes[current_pos + i + 1]
                    
                    # Get valid tickers for prediction
                    valid_tickers = self.preprocessor.get_valid_tickers_for_prediction(episode)
                    
                    if valid_tickers:
                        # Create features for prediction
                        episode_data = [episode]
                        X_pred, _, metadata_pred = self.preprocessor.transform(episode_data)
                        
                        if len(X_pred) > 0:
                            # Filter to valid tickers
                            valid_indices = [j for j, meta in enumerate(metadata_pred) 
                                           if meta['ticker'] in valid_tickers]
                            
                            if valid_indices:
                                X_pred_filtered = X_pred[valid_indices]
                                tickers_filtered = [metadata_pred[j]['ticker'] for j in valid_indices]
                                
                                # Get predictions
                                predictions = self.model.predict_returns(
                                    torch.FloatTensor(X_pred_filtered).to(self.device),
                                    tickers_filtered
                                )
                                
                                # Get actual returns
                                for ticker in tickers_filtered:
                                    pred_return = predictions[ticker]
                                    actual_return = self.preprocessor.calculate_ticker_return(
                                        episode, next_episode, ticker
                                    )
                                    
                                    if actual_return is not None:
                                        results['predictions'].append(pred_return)
                                        results['actuals'].append(actual_return)
                                        results['dates'].append(episode.date)
                                        results['tickers'].append(ticker)
            
            current_pos = end_pos
        
        # Calculate overall metrics
        if results['predictions']:
            predictions = np.array(results['predictions'])
            actuals = np.array(results['actuals'])
            
            final_metrics = {
                'mse': mean_squared_error(actuals, predictions),
                'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                'mae': mean_absolute_error(actuals, predictions),
                'r2': r2_score(actuals, predictions),
                'num_predictions': len(predictions),
                'num_retrains': len(results['retrain_dates'])
            }
            
            results['final_metrics'] = final_metrics
            
            logger.info(f"Walk-forward validation completed: {final_metrics}")
        else:
            logger.warning("No predictions made during walk-forward validation")
        
        return results
    
    def save_training_plots(self, save_dir: str = "training_plots"):
        """Save training history plots."""
        Path(save_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        axes[0, 1].plot(self.history['train_mae'], label='Train MAE')
        axes[0, 1].plot(self.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # R2 plot
        axes[1, 0].plot(self.history['train_r2'], label='Train R²')
        axes[1, 0].plot(self.history['val_r2'], label='Val R²')
        axes[1, 0].set_title('R² Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.history['learning_rates'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {save_dir}/training_history.png")