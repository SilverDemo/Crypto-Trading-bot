import json
from datetime import datetime
from threading import Lock

class Reporter:
    def __init__(self):
        self.trades = []
        self.portfolio = []
        self.lock = Lock()
    
    def log_trade(self, symbol, action, quantity, price):
        with self.lock:
            self.trades.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'BUY' if action > 0 else 'SELL',
                'quantity': quantity,
                'price': price
            })
            self._save('trades.json', {'trades': self.trades[-100:]})
    
    def update_portfolio(self, value, positions):
        with self.lock:
            self.portfolio.append({
                'timestamp': datetime.now().isoformat(),
                'value': value,
                'positions': positions
            })
            self._save('portfolio.json', {
                'history': self.portfolio[-1440:],
                'current': self.portfolio[-1]
            })
    
    def _save(self, filename, data):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
