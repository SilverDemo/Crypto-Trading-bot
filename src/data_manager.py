import ccxt
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
from configparser import ConfigParser

class DataManager:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read('config.ini')
        self._init_exchange()
        self._load_initial_data()
        
    def _init_exchange(self):
        self.exchange = ccxt.binance({
            'apiKey': self.config.get('binance', 'api_key'),
            'secret': self.config.get('binance', 'api_secret'),
            'enableRateLimit': True,
            'options': {
                'test': self.config.getboolean('binance', 'testnet'),
                'adjustForTimeDifference': True
            }
        })

    def _fetch(self, symbol, start_date, limit):
        since = self._datetime_to_ms(start_date)
        timeframe = '1m'
        all_data = []
        remaining = limit
        current_since = since
        
        while remaining is None or remaining > 0:
            try:
                # Calculate chunk size
                chunk_limit = min(remaining, 1000) if remaining else None
                
                # Fetch data
                data = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=chunk_limit
                )
                
                if not data:
                    break
                
                # Store and update pointers
                all_data.extend(data)
                if remaining is not None:
                    remaining -= len(data)
                
                # Update since for next request
                current_since = data[-1][0] + self._timeframe_to_ms(timeframe)
                
                # Rate limit protection
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error fetching data: {str(e)}")
                break
                
        return all_data
        
    def _load_initial_data(self):
        self.mode = self.config.get('model', 'mode')
        self.symbols = self.config.get('model', 'symbols').split(',')
        self.window_size = self.config.getint('data', 'window_size')
        self.data = {}
        
        if self.mode == 'training':
            start_date = pd.to_datetime(self.config.get('training_param', 'start_date'))
            for symbol in self.symbols:
                self.data[symbol] = self._fetch_historical(symbol, start_date)
        else:
            for symbol in self.symbols:
                self.data[symbol] = self._fetch_recent(symbol)
                
    def _fetch_historical(self, symbol, start_date):
        timeframe = '1m'
        since = self._datetime_to_ms(start_date)
        all_data = self._fetch(symbol, start_date, self.window_size)
        
        return self._process_data(all_data[-self.window_size:], symbol)
    
    def _fetch_recent(self, symbol):
        data = self._fetch(symbol, start_date, self.window_size)
        return self._process_data(data, symbol)
    
    def _process_data(self, data, symbol):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Improved normalization with Z-score
        means = df.mean()
        stds = df.std()
        stds = stds.replace(0, 1e-8)  # Prevent division by zero
        
        df_norm = (df - means) / stds
        df_norm = df_norm.clip(-5, 5)  # Clip extreme values
        return df_norm
    
    def update_training_data(self):
        for symbol in self.symbols:
            self.data[symbol] = self.data[symbol].iloc[1:].append(
                self._fetch_next(symbol)
            )
    
    def _fetch_next(self, symbol):
        last_ts = self.data[symbol].index[-1].timestamp() * 1000
        data = self.exchange.fetch_ohlcv(
            symbol, '1m', since=last_ts + 60000, limit=1
        )
        return self._process_data(data, symbol)
    
    def update_live_data(self):
        for symbol in self.symbols:
            new_data = self._fetch_recent(symbol)
            self.data[symbol] = pd.concat([self.data[symbol], new_data]).iloc[-self.window_size:]
    
    def get_window(self, symbol):
        # Ensure we return (1, 1440, 5) shape
        data = self.data[symbol].iloc[-self.window_size:]
        if len(data) < self.window_size:
            # Pad with zeros if insufficient data
            pad_size = self.window_size - len(data)
            pad_df = pd.DataFrame(np.zeros((pad_size, 5)), 
                                columns=['open', 'high', 'low', 'close', 'volume'])
            data = pd.concat([pad_df, data])
        return data.values.reshape(1, self.window_size, 5)
    
    def _datetime_to_ms(self, dt):
        return int(dt.timestamp() * 1000)

    def _timeframe_to_ms(self, timeframe):
        """Convert timeframe string to milliseconds"""
        units = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000
        }
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * units[unit]