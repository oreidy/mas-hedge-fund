"""
LSTM-based Reinforcement Learning module for hedge fund trading decisions.

This module implements a Partially Observable Markov Decision Process (POMDP) using
LSTM-based recurrent reinforcement learning to handle market uncertainties and refine
LLM trading decisions.
"""

from .data_collection import TrainingEpisode, EpisodeCollector
from .preprocessing import DataPreprocessor, FeatureExtractor
from .models import LSTMTradingModel
from .training import RLTrainer
from .inference import RLInference

__all__ = [
    'TrainingEpisode',
    'EpisodeCollector', 
    'DataPreprocessor',
    'FeatureExtractor',
    'LSTMTradingModel',
    'RLTrainer',
    'RLInference'
]