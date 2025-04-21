# Crypto Trading Bot

A reinforcement learning-based crypto trading bot with real-time reporting.

## Features
- Binance integration
- Minute-by-minute trading decisions
- Real-time JSON reporting
- Auto-updating HTML dashboard
- PPO-based AI model

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Configure config.ini with your Binance API keys 
Example:
```config
[binance]
api_key = YOUR_API_KEY
api_secret = YOUR_API_SECRET
testnet = False  # Set to True for testing

[model]
mode = training  # training or live
symbols = BTC/USDT
path = trained_model.zip
initial_balance = 10000.0

[data]
start_date = 2023-01-01 00:00:00
window_size = 1440  # 24h in minutes

[trading]
trade_fee = 0.001  # 0.1%
max_position = 0.9  # 90% of portfolio
```

3. Train the model:
```bash
python -m src.model_trainer
```
4. Run the bot:

```bash
python -m src.bot
```
5. Open templates/dashboard.html in your browser


**4. requirements.txt**
