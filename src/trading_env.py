import gymnasium as gym
import numpy as np
from gymnasium import spaces
from configparser import ConfigParser

class TradingEnvironment(gym.Env):
    def __init__(self, data_manager):
        super().__init__()
        self.config = ConfigParser()
        self.config.read('config.ini')
        
        self.data_manager = data_manager
        self.symbols = self.config.get('model', 'symbols').split(',')
        self.window_size = self.config.getint('data', 'window_size')
        self.current_step = 0
        self.prev_value = 0.0
        self.trade_fee = self.config.getfloat('trading', 'trade_fee')

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(self.symbols),), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(len(self.symbols), self.window_size, 5),
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        self.balance = self.config.getfloat('model', 'initial_balance')
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.current_step = 0
        self.prev_value = self.portfolio_value
        return self._get_observation(), {}
    
    def step(self, actions):
        self.current_step += 1
        self._execute_trades(actions)
        
        reward = self._calculate_reward()
        done = self.current_step >= len(self.data_manager.data[self.symbols[0]]) - 1
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        obs = []
        for symbol in self.symbols:
            window = self.data_manager.get_window(symbol)
            # Ensure correct shape
            if window.shape != (1, self.window_size, 5):
                raise ValueError(f"Invalid window shape {window.shape} for {symbol}")
            obs.append(window)
        return np.concatenate(obs, axis=0)
        
    def _execute_trades(self, actions):
        current_prices = self._get_current_prices()
        total_value = self.portfolio_value
        
        for i, symbol in enumerate(self.symbols):
            action = actions[i]
            price = current_prices[symbol]
            
            if action > 0:  # Buy
                buy_amount = action * total_value
                fee = buy_amount * self.trade_fee
                
                # Deduct fee from buy amount
                self.positions[symbol] += (buy_amount - fee) / price
                self.balance -= buy_amount  # Full amount deducted (includes fee)
                
            elif action < 0:  # Sell
                sell_fraction = abs(action)
                sell_quantity = self.positions[symbol] * sell_fraction
                sale_value = sell_quantity * price
                fee = sale_value * self.trade_fee
                
                # Deduct fee from sale proceeds
                self.positions[symbol] -= sell_quantity
                self.balance += sale_value - fee
    
    def _get_current_prices(self):
        return {
            symbol: self.data_manager.data[symbol]['close'].iloc[self.current_step]
            for symbol in self.symbols
        }
    
    def _calculate_reward(self):
        new_value = self.portfolio_value
        reward = new_value - self.prev_value  # Simple absolute reward
        self.prev_value = new_value
        return np.clip(reward, -1, 1)
    
    @property
    def portfolio_value(self):
        prices = self._get_current_prices()
        return self.balance + sum(
            qty * prices[symbol] for symbol, qty in self.positions.items()
        )
    
    @property
    def initial_value(self):
        return self.config.getfloat('model', 'initial_balance')