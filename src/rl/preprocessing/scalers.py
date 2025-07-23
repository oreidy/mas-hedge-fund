"""
Scaling utilities for RL training.

This module provides utilities for scaling features and targets
for LSTM training, ensuring proper normalization and handling
of different types of features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
from pathlib import Path

from utils.logger import logger


def create_scalers(
    features_df: pd.DataFrame,
    feature_scaler: str = "standard",
    target_scaler: str = "minmax",
    exclude_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create and fit scalers for features and targets.
    
    Args:
        features_df: DataFrame with features
        feature_scaler: Type of scaler for features ('standard', 'minmax', 'robust')
        target_scaler: Type of scaler for targets ('standard', 'minmax', 'robust')
        exclude_columns: Columns to exclude from scaling
        
    Returns:
        Dictionary containing fitted scalers
    """
    if exclude_columns is None:
        exclude_columns = ['date', 'portfolio_return']
    
    scalers = {}
    
    # Feature columns (exclude date and target)
    feature_columns = [col for col in features_df.columns if col not in exclude_columns]
    
    # Create feature scaler
    if feature_scaler == "standard":
        scalers['features'] = StandardScaler()
    elif feature_scaler == "minmax":
        scalers['features'] = MinMaxScaler()
    elif feature_scaler == "robust":
        scalers['features'] = RobustScaler()
    else:
        raise ValueError(f"Unsupported feature scaler: {feature_scaler}")
    
    # Fit feature scaler
    if feature_columns:
        feature_data = features_df[feature_columns].values
        scalers['features'].fit(feature_data)
        logger.info(f"Fitted {feature_scaler} scaler for {len(feature_columns)} features")
    
    # Create target scaler
    if 'portfolio_return' in features_df.columns:
        if target_scaler == "standard":
            scalers['targets'] = StandardScaler()
        elif target_scaler == "minmax":
            scalers['targets'] = MinMaxScaler()
        elif target_scaler == "robust":
            scalers['targets'] = RobustScaler()
        else:
            raise ValueError(f"Unsupported target scaler: {target_scaler}")
        
        # Fit target scaler
        target_data = features_df[['portfolio_return']].values
        scalers['targets'].fit(target_data)
        logger.info(f"Fitted {target_scaler} scaler for targets")
    
    # Store metadata
    scalers['feature_columns'] = feature_columns
    scalers['feature_scaler_type'] = feature_scaler
    scalers['target_scaler_type'] = target_scaler
    
    return scalers


def apply_scalers(
    features_df: pd.DataFrame,
    scalers: Dict[str, Any],
    exclude_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply fitted scalers to features and targets.
    
    Args:
        features_df: DataFrame with features
        scalers: Dictionary with fitted scalers
        exclude_columns: Columns to exclude from scaling
        
    Returns:
        DataFrame with scaled features
    """
    if exclude_columns is None:
        exclude_columns = ['date', 'portfolio_return']
    
    # Make a copy to avoid modifying the original
    scaled_df = features_df.copy()
    
    # Scale features
    feature_columns = scalers.get('feature_columns', [])
    if feature_columns and 'features' in scalers:
        # Ensure all feature columns exist in the DataFrame
        available_columns = [col for col in feature_columns if col in scaled_df.columns]
        
        if available_columns:
            feature_data = scaled_df[available_columns].values
            scaled_features = scalers['features'].transform(feature_data)
            scaled_df[available_columns] = scaled_features
            logger.debug(f"Scaled {len(available_columns)} feature columns")
    
    # Scale targets
    if 'portfolio_return' in scaled_df.columns and 'targets' in scalers:
        target_data = scaled_df[['portfolio_return']].values
        scaled_targets = scalers['targets'].transform(target_data)
        scaled_df['portfolio_return'] = scaled_targets.flatten()
        logger.debug("Scaled target column")
    
    return scaled_df


def inverse_transform_targets(
    scaled_targets: np.ndarray,
    scalers: Dict[str, Any]
) -> np.ndarray:
    """
    Inverse transform scaled targets back to original scale.
    
    Args:
        scaled_targets: Scaled target values
        scalers: Dictionary with fitted scalers
        
    Returns:
        Targets in original scale
    """
    if 'targets' not in scalers:
        logger.warning("No target scaler found, returning targets as-is")
        return scaled_targets
    
    # Reshape for sklearn if needed
    if scaled_targets.ndim == 1:
        scaled_targets = scaled_targets.reshape(-1, 1)
    
    original_targets = scalers['targets'].inverse_transform(scaled_targets)
    
    return original_targets.flatten() if original_targets.shape[1] == 1 else original_targets


def create_robust_scalers(
    features_df: pd.DataFrame,
    feature_percentiles: Tuple[float, float] = (5.0, 95.0),
    target_percentiles: Tuple[float, float] = (1.0, 99.0),
    exclude_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create robust scalers that handle outliers better.
    
    Args:
        features_df: DataFrame with features
        feature_percentiles: Percentiles for feature clipping
        target_percentiles: Percentiles for target clipping
        exclude_columns: Columns to exclude from scaling
        
    Returns:
        Dictionary containing fitted scalers
    """
    if exclude_columns is None:
        exclude_columns = ['date', 'portfolio_return']
    
    scalers = {}
    
    # Feature columns
    feature_columns = [col for col in features_df.columns if col not in exclude_columns]
    
    if feature_columns:
        feature_data = features_df[feature_columns].values
        
        # Calculate percentiles for clipping
        lower_percentile, upper_percentile = feature_percentiles
        lower_bounds = np.percentile(feature_data, lower_percentile, axis=0)
        upper_bounds = np.percentile(feature_data, upper_percentile, axis=0)
        
        # Clip outliers
        clipped_data = np.clip(feature_data, lower_bounds, upper_bounds)
        
        # Create and fit robust scaler
        scaler = RobustScaler()
        scaler.fit(clipped_data)
        
        scalers['features'] = scaler
        scalers['feature_bounds'] = (lower_bounds, upper_bounds)
        
        logger.info(f"Fitted robust scaler for {len(feature_columns)} features with "
                   f"{lower_percentile}-{upper_percentile} percentile clipping")
    
    # Target scaling
    if 'portfolio_return' in features_df.columns:
        target_data = features_df[['portfolio_return']].values
        
        # Calculate percentiles for clipping
        lower_percentile, upper_percentile = target_percentiles
        lower_bound = np.percentile(target_data, lower_percentile)
        upper_bound = np.percentile(target_data, upper_percentile)
        
        # Clip outliers
        clipped_targets = np.clip(target_data, lower_bound, upper_bound)
        
        # Create and fit robust scaler
        scaler = RobustScaler()
        scaler.fit(clipped_targets)
        
        scalers['targets'] = scaler
        scalers['target_bounds'] = (lower_bound, upper_bound)
        
        logger.info(f"Fitted robust target scaler with "
                   f"{lower_percentile}-{upper_percentile} percentile clipping")
    
    # Store metadata
    scalers['feature_columns'] = feature_columns
    scalers['feature_scaler_type'] = 'robust'
    scalers['target_scaler_type'] = 'robust'
    scalers['feature_percentiles'] = feature_percentiles
    scalers['target_percentiles'] = target_percentiles
    
    return scalers


def apply_robust_scalers(
    features_df: pd.DataFrame,
    scalers: Dict[str, Any],
    exclude_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Apply robust scalers with outlier clipping.
    
    Args:
        features_df: DataFrame with features
        scalers: Dictionary with fitted scalers
        exclude_columns: Columns to exclude from scaling
        
    Returns:
        DataFrame with scaled features
    """
    if exclude_columns is None:
        exclude_columns = ['date', 'portfolio_return']
    
    # Make a copy to avoid modifying the original
    scaled_df = features_df.copy()
    
    # Scale features with outlier clipping
    feature_columns = scalers.get('feature_columns', [])
    if feature_columns and 'features' in scalers:
        available_columns = [col for col in feature_columns if col in scaled_df.columns]
        
        if available_columns:
            feature_data = scaled_df[available_columns].values
            
            # Apply outlier clipping if bounds are available
            if 'feature_bounds' in scalers:
                lower_bounds, upper_bounds = scalers['feature_bounds']
                feature_data = np.clip(feature_data, lower_bounds, upper_bounds)
            
            scaled_features = scalers['features'].transform(feature_data)
            scaled_df[available_columns] = scaled_features
            logger.debug(f"Scaled {len(available_columns)} feature columns with robust scaling")
    
    # Scale targets with outlier clipping
    if 'portfolio_return' in scaled_df.columns and 'targets' in scalers:
        target_data = scaled_df[['portfolio_return']].values
        
        # Apply outlier clipping if bounds are available
        if 'target_bounds' in scalers:
            lower_bound, upper_bound = scalers['target_bounds']
            target_data = np.clip(target_data, lower_bound, upper_bound)
        
        scaled_targets = scalers['targets'].transform(target_data)
        scaled_df['portfolio_return'] = scaled_targets.flatten()
        logger.debug("Scaled target column with robust scaling")
    
    return scaled_df


def save_scalers(scalers: Dict[str, Any], filepath: Path) -> None:
    """
    Save fitted scalers to disk.
    
    Args:
        scalers: Dictionary with fitted scalers
        filepath: Path to save the scalers
    """
    with open(filepath, 'wb') as f:
        pickle.dump(scalers, f)
    
    logger.info(f"Scalers saved to {filepath}")


def load_scalers(filepath: Path) -> Dict[str, Any]:
    """
    Load fitted scalers from disk.
    
    Args:
        filepath: Path to load the scalers from
        
    Returns:
        Dictionary with fitted scalers
    """
    with open(filepath, 'rb') as f:
        scalers = pickle.load(f)
    
    logger.info(f"Scalers loaded from {filepath}")
    
    return scalers


def validate_scalers(scalers: Dict[str, Any]) -> bool:
    """
    Validate that scalers are properly configured.
    
    Args:
        scalers: Dictionary with scalers
        
    Returns:
        True if scalers are valid, False otherwise
    """
    required_keys = ['feature_columns', 'feature_scaler_type']
    
    for key in required_keys:
        if key not in scalers:
            logger.error(f"Missing required key in scalers: {key}")
            return False
    
    if 'features' not in scalers:
        logger.error("No feature scaler found")
        return False
    
    # Check if feature scaler is fitted
    if not hasattr(scalers['features'], 'scale_'):
        logger.error("Feature scaler is not fitted")
        return False
    
    logger.info("Scalers validation passed")
    return True


def get_scaling_stats(scalers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about the fitted scalers.
    
    Args:
        scalers: Dictionary with fitted scalers
        
    Returns:
        Dictionary with scaling statistics
    """
    stats = {}
    
    # Feature scaler stats
    if 'features' in scalers:
        feature_scaler = scalers['features']
        stats['feature_scaler_type'] = scalers.get('feature_scaler_type', 'unknown')
        stats['num_features'] = len(scalers.get('feature_columns', []))
        
        if hasattr(feature_scaler, 'scale_'):
            stats['feature_scale_mean'] = float(np.mean(feature_scaler.scale_))
            stats['feature_scale_std'] = float(np.std(feature_scaler.scale_))
        
        if hasattr(feature_scaler, 'mean_'):
            stats['feature_mean_mean'] = float(np.mean(feature_scaler.mean_))
            stats['feature_mean_std'] = float(np.std(feature_scaler.mean_))
    
    # Target scaler stats
    if 'targets' in scalers:
        target_scaler = scalers['targets']
        stats['target_scaler_type'] = scalers.get('target_scaler_type', 'unknown')
        
        if hasattr(target_scaler, 'scale_'):
            stats['target_scale'] = float(target_scaler.scale_[0])
        
        if hasattr(target_scaler, 'mean_'):
            stats['target_mean'] = float(target_scaler.mean_[0])
    
    return stats