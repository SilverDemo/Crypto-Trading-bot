from stable_baselines3 import PPO
from .data_manager import DataManager
from .trading_env import TradingEnvironment
from configparser import ConfigParser
import time

class TradingBot:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read('config.ini')
        
        self.data_manager = DataManager()
        self.model = PPO.load(self.config.get('model', 'path'))
        self.env = TradingEnvironment(self.data_manager)
    
    def run(self):
        while True:
            self.data_manager.update_live_data()
            obs = self.env.reset()[0]
            action, _ = self.model.predict(obs)
            self._execute_real_trades(action)
            time.sleep(60)
    
    def _execute_real_trades(self, actions):
        exchange = self.data_manager.exchange
        for i, symbol in enumerate(self.env.symbols):
            action = actions[i]
            if action > 0:
                exchange.create_market_buy_order(symbol, self._calculate_size(action))
            elif action < 0:
                exchange.create_market_sell_order(symbol, self._calculate_size(abs(action)))
    
    def _calculate_size(self, action):
        balance = self.config.getfloat('model', 'initial_balance')
        return action * balance * self.config.getfloat('trading', 'max_position')

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()