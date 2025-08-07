#!/usr/bin/env python3
"""
Generate daily returns CSV using trained LSTM model for benchmarking.

This script loads the trained LSTM model and generates daily portfolio returns
in the same format as your existing benchmarking data.
"""

import sys
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path for imports
sys.path.append('src')

from rl.data_collection.episode import TrainingEpisode, AgentSignal, LLMDecision, PortfolioState
from rl.preprocessing.preprocessor import DataPreprocessor
from rl.models.lstm_model import LSTMTradingModel

def load_all_episodes() -> list:
    """Load all training episodes in chronological order."""
    episodes_dir = Path("training_data/episodes")
    json_files = sorted(list(episodes_dir.glob("*.json")))
    
    episodes = []
    print(f"Loading {len(json_files)} episodes...")
    
    for json_file in tqdm(json_files, desc="Loading episodes"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Convert agent signals to proper format
            processed_agent_signals = {}
            for agent_name, signals in data['agent_signals'].items():
                processed_signals = {}
                for ticker, signal_data in signals.items():
                    if isinstance(signal_data, dict) and 'signal' in signal_data and 'confidence' in signal_data:
                        processed_signals[ticker] = AgentSignal.from_dict(signal_data)
                
                if processed_signals:
                    processed_agent_signals[agent_name] = processed_signals
            
            episode = TrainingEpisode(
                date=data['date'],
                agent_signals=processed_agent_signals,
                llm_decisions={
                    ticker: LLMDecision.from_dict(decision_data)
                    for ticker, decision_data in data['llm_decisions'].items()
                },
                portfolio_before=PortfolioState.from_dict(data['portfolio_before']),
                portfolio_after=PortfolioState.from_dict(data['portfolio_after']),
                portfolio_return=data['portfolio_return'],
                market_data=data.get('market_data', {}),
                metadata=data.get('metadata', {})
            )
            episodes.append(episode)
            
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(episodes)} episodes")
    return episodes

def load_trained_model() -> tuple:
    """Load the trained LSTM model and preprocessor."""
    model_path = Path("models/lstm/lstm_trading_model.pt")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    print(f"Loading trained model from {model_path}")
    model = LSTMTradingModel.load_model(model_path, device='cpu')
    
    # Create preprocessor with same parameters used in training
    preprocessor = DataPreprocessor(
        sequence_length=20,
        prediction_horizon=1,
        feature_scaler="standard",
        include_market_features=True,
        include_consensus_features=True
    )
    
    return model, preprocessor

def generate_lstm_predictions(episodes: list, model: LSTMTradingModel, preprocessor: DataPreprocessor) -> dict:
    """Generate LSTM predictions for all episodes."""
    print("Generating LSTM predictions...")
    
    # Fit preprocessor on all episodes (as done in training)
    preprocessor.fit(episodes)
    
    # Transform to sequences
    processed_data = preprocessor.transform(episodes)
    sequences = processed_data['sequences']
    
    print(f"Generated {sequences.shape[0]} sequences for prediction")
    
    # Convert to tensor
    X = torch.FloatTensor(sequences)
    
    # Generate predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            batch_preds = model(batch).squeeze().cpu().numpy()
            
            # Handle single prediction case
            if batch_preds.ndim == 0:
                batch_preds = [batch_preds.item()]
            elif batch_preds.ndim == 1:
                batch_preds = batch_preds.tolist()
            
            predictions.extend(batch_preds)
    
    print(f"Generated {len(predictions)} LSTM predictions")
    
    # Map predictions back to episodes (skip first sequence_length episodes)
    prediction_episodes = episodes[preprocessor.sequence_length:]
    
    results = {}
    for i, episode in enumerate(prediction_episodes):
        if i < len(predictions):
            results[episode.date] = {
                'lstm_predicted_return': predictions[i],
                'actual_return': episode.portfolio_return,
                'date': episode.date
            }
    
    return results

def create_returns_csv(predictions: dict, episodes: list) -> pd.DataFrame:
    """Create daily returns CSV in benchmarking format."""
    print("Creating returns CSV...")
    
    # Create a comprehensive returns dataframe
    data = []
    
    for episode in episodes:
        date = episode.date
        actual_return = episode.portfolio_return
        
        # Get LSTM prediction if available
        lstm_prediction = predictions.get(date, {}).get('lstm_predicted_return', np.nan)
        
        data.append({
            'date': date,
            'actual_return': actual_return,
            'lstm_predicted_return': lstm_prediction,
            'portfolio_value_before': episode.portfolio_before.cash,
            'portfolio_value_after': episode.portfolio_after.cash
        })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate cumulative returns for benchmarking
    df['actual_cumulative_return'] = (1 + df['actual_return']).cumprod()
    
    # For LSTM predictions where available, calculate cumulative returns
    lstm_returns = df['lstm_predicted_return'].fillna(0)  # Fill NaN with 0 for missing predictions
    df['lstm_cumulative_return'] = (1 + lstm_returns).cumprod()
    
    # Calculate performance metrics
    actual_total_return = df['actual_cumulative_return'].iloc[-1] - 1
    lstm_total_return = df['lstm_cumulative_return'].iloc[-1] - 1
    
    actual_volatility = df['actual_return'].std() * np.sqrt(252)  # Annualized volatility
    lstm_volatility = lstm_returns.std() * np.sqrt(252)
    
    actual_sharpe = (actual_total_return / len(df) * 252) / actual_volatility if actual_volatility > 0 else 0
    lstm_sharpe = (lstm_total_return / len(df) * 252) / lstm_volatility if lstm_volatility > 0 else 0
    
    print(f"\n📊 Performance Summary:")
    print(f"  - Total episodes: {len(df)}")
    print(f"  - Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  - Actual total return: {actual_total_return:.2%}")
    print(f"  - LSTM predicted total return: {lstm_total_return:.2%}")
    print(f"  - Actual volatility (annualized): {actual_volatility:.2%}")
    print(f"  - LSTM volatility (annualized): {lstm_volatility:.2%}")
    print(f"  - Actual Sharpe ratio: {actual_sharpe:.3f}")
    print(f"  - LSTM Sharpe ratio: {lstm_sharpe:.3f}")
    
    return df

def main():
    """Main function to generate LSTM returns CSV."""
    print("🚀 Generating LSTM trading returns for benchmarking...")
    
    # Load episodes
    episodes = load_all_episodes()
    
    if len(episodes) < 50:
        print(f"❌ Not enough episodes: {len(episodes)}")
        return
    
    # Load trained model
    try:
        model, preprocessor = load_trained_model()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run train_lstm.py first to train the model.")
        return
    
    # Generate predictions
    predictions = generate_lstm_predictions(episodes, model, preprocessor)
    
    # Create returns CSV
    returns_df = create_returns_csv(predictions, episodes)
    
    # Save to benchmarking directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmarking/data/lstm_returns_{timestamp}.csv"
    
    # Ensure directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with same format as existing benchmarking files
    final_df = returns_df[['date', 'actual_return']].rename(columns={'actual_return': 'return'})
    final_df.to_csv(output_file, index=False)
    
    # Also save detailed version with LSTM predictions
    detailed_file = f"benchmarking/data/lstm_detailed_returns_{timestamp}.csv"
    returns_df.to_csv(detailed_file, index=False)
    
    print(f"\n✅ Returns CSV saved:")
    print(f"  - Benchmarking format: {output_file}")
    print(f"  - Detailed analysis: {detailed_file}")
    print(f"\nYou can now use these files with your existing analyze_performance.py script!")

if __name__ == "__main__":
    main()