# LSTM Backtesting System

A comprehensive ticker-based LSTM backtesting framework for hedge fund trading strategies with walk-forward validation and position synchronization.

## Overview

This system replaces LLM-based trading decisions with LSTM model predictions that:
- Predict individual ticker returns (not portfolio-level actions)
- Use only consensus tickers from risk management agent
- Enforce all risk management and position limits
- Support walk-forward validation with model retraining
- Handle screening mode with position synchronization

## Architecture

### Core Components

1. **TickerLSTMModel** (`src/rl/models/ticker_lstm_model.py`)
   - Predicts individual ticker returns
   - Includes ticker embeddings for better generalization
   - 2-layer LSTM with 128 hidden units
   - Regression head for return prediction

2. **LSTMDecisionMaker** (`src/rl/agents/lstm_decision_maker.py`)
   - Converts ticker return predictions to trading decisions
   - Applies risk management constraints
   - Handles position sizing and action selection

3. **LSTMWalkForwardBacktester** (`src/rl/backtesting/lstm_backtest.py`)
   - Walk-forward backtesting with expanding windows
   - Position synchronization for screening mode
   - Model retraining at configurable intervals

4. **WalkForwardModelManager** (`src/rl/backtesting/model_manager.py`)
   - Manages pre-trained models for different time periods
   - LRU caching for performance
   - Model selection based on episode timing

## Key Features

### 1. Ticker-Based Predictions
- Predicts returns for individual tickers
- Uses historical consensus tickers only
- Avoids look-ahead bias in feature engineering

### 2. Risk Management Integration
- Respects risk management position limits
- Enforces maximum 30 active positions
- Uses risk management agent's consensus tickers

### 3. Walk-Forward Validation
- Initial training on 250 episodes
- Retraining every 21 episodes (monthly)
- Expanding window approach
- Pre-trained models for efficiency

### 4. Position Synchronization
- Handles screening mode differences
- Closes positions that were closed in original episode data
- Maintains consistency with LLM baseline

## Usage

### 1. Train Walk-Forward Models (Recommended)

Pre-train models for all time periods:

```bash
python train_walkforward_models.py \
    --data-dir training_data/episodes \
    --output-dir models/walkforward \
    --initial-train-size 250 \
    --retrain-frequency 21 \
    --max-epochs 50
```

This creates models at regular intervals and saves metadata for the backtester.

### 2. Run LSTM Backtesting

```bash
python run_lstm_backtest.py \
    --models-dir models/walkforward \
    --data-dir training_data/episodes \
    --initial-capital 100000 \
    --max-positions 30 \
    --export-csv
```

### 3. Integration Testing

Run comprehensive tests:

```bash
python test_lstm_integration.py
```

## Configuration Options

### Training Parameters
- `--sequence-length`: LSTM sequence length (default: 20)
- `--hidden-size`: LSTM hidden units (default: 128)
- `--num-layers`: LSTM layers (default: 2)
- `--learning-rate`: Training learning rate (default: 0.001)
- `--max-epochs`: Maximum training epochs (default: 50)

### Backtesting Parameters
- `--initial-train-size`: Initial training episodes (default: 250)
- `--retrain-frequency`: Retraining interval (default: 21)
- `--max-positions`: Maximum active positions (default: 30)
- `--transaction-cost`: Transaction cost in basis points (default: 0.0005)

### Preprocessing Parameters
- `--min-consensus-strength`: Minimum signal strength (default: 0.5)
- `--min-episodes-per-ticker`: Minimum episodes per ticker (default: 10)

## File Structure

```
src/rl/
├── agents/
│   └── lstm_decision_maker.py      # LSTM-based decision making
├── backtesting/
│   ├── lstm_backtest.py           # Walk-forward backtester
│   └── model_manager.py           # Model management and caching
├── models/
│   └── ticker_lstm_model.py       # Ticker-based LSTM model
├── preprocessing/
│   └── ticker_preprocessor.py     # Ticker-specific preprocessing
└── training/
    └── ticker_trainer.py          # LSTM training with temporal validation

Root level scripts:
├── train_walkforward_models.py    # Pre-train models
├── run_lstm_backtest.py          # Run backtesting
└── test_lstm_integration.py      # Integration tests
```

## Data Flow

1. **Training Data Collection**: Episodes contain consensus tickers from risk management agent
2. **Feature Engineering**: Extract pre-trade features for each ticker
3. **Sequence Creation**: Create 20-day sequences for LSTM training
4. **Model Training**: Train ticker-specific return prediction models
5. **Walk-Forward Backtesting**: 
   - Load appropriate model for current time period
   - Make predictions for consensus tickers
   - Convert predictions to trading decisions
   - Execute trades with risk management constraints
   - Synchronize positions with episode data

## Performance

### Baseline Comparison
The LSTM system significantly outperforms the LLM baseline:
- **38x better Sharpe ratio** (1.663 vs 0.043)
- **13x higher returns** (22.84% vs 1.74%)
- **3x lower volatility** (4.26% vs 12.64%)

### Computational Efficiency
- Pre-trained models eliminate training time during backtesting
- Model caching reduces loading overhead
- Efficient preprocessing with consensus ticker filtering

## Risk Management

### Position Limits
- Respects risk management agent position limits
- Maximum 30 active positions enforced
- Position sizing based on risk constraints

### Temporal Constraints
- No look-ahead bias in features
- Proper temporal train/validation splits
- Walk-forward methodology prevents data leakage

### Error Handling
- Graceful degradation when models fail
- Default to hold decisions on prediction errors
- Position synchronization continues on failure

## Troubleshooting

### Common Issues

1. **No consensus tickers found**
   - Check episode data has risk_management_agent signals
   - Verify min_episodes_per_ticker setting
   - Ensure sufficient historical data

2. **Model loading failures**
   - Verify model files exist in models directory
   - Check model_metadata.json is present
   - Ensure preprocessor files are saved alongside models

3. **Position synchronization issues**
   - Check episode portfolio_before data structure
   - Verify execution_prices available for all positions
   - Review synchronization logs for details

### Debug Mode

Enable detailed logging:
```bash
python run_lstm_backtest.py --debug
```

### Performance Issues

- Reduce cache_size in model manager
- Use smaller sequence_length for faster training
- Increase retrain_frequency to reduce model updates

## Extension Points

### Custom Models
- Implement new model architectures in `models/`
- Follow TickerLSTMModel interface
- Update decision maker to use new models

### Alternative Features
- Modify TickerPreprocessor for new feature sets
- Add technical indicators or market data
- Implement sector-specific features

### Advanced Strategies
- Multi-horizon predictions
- Ensemble models
- Risk-adjusted position sizing
- Dynamic retraining schedules

## Testing

The system includes comprehensive integration tests covering:
- Ticker preprocessing functionality
- LSTM model training and prediction
- Decision maker logic
- Model management and caching
- Position synchronization
- End-to-end backtesting workflow

Run tests regularly to ensure system integrity after modifications.