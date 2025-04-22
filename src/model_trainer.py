from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from .trading_env import TradingEnvironment
from .data_manager import DataManager
import configparser
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
class TimeEstimateCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.formatter = lambda x: f"{x:02d}"

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.start_time is None:
            return True

        current_progress = self.num_timesteps / self.total_timesteps
        elapsed_time = time.time() - self.start_time
        
        if current_progress > 0:
            estimated_total = elapsed_time / current_progress
            remaining = estimated_total - elapsed_time
        else:
            remaining = float('inf')

        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(remaining) if remaining != float('inf') else "--:--:--"
        
        print(f"\rProgress: {current_progress*100:.1f}% ({self.num_timesteps}/{self.total_timesteps})| "
                  f"Elapsed: {elapsed_str} | "
                  f"Remaining: {remaining_str} | "
                  f"Steps/s: {self.num_timesteps/elapsed_time:.1f}",
                  end="", flush=True)
        return True

    def _format_time(self, seconds):
        if seconds == float('inf'):
            return "--:--:--"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{self.formatter(hours)}:{self.formatter(minutes)}:{self.formatter(seconds)}"


class ModelTrainer:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        
        self.data_manager = DataManager()
        
        # Create vectorized environment
        self.env = DummyVecEnv([lambda: Monitor(TradingEnvironment(self.data_manager))])
        
        # Add normalization
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)
        self.total_timesteps = int(self.config.get('training_param', 'total_timesteps'))
        self.path = self.config.get('model', 'path')
        if not os.path.exists(self.path):
            self.model = PPO(
                'MlpPolicy',
                self.env,
                verbose=1,
                tensorboard_log='./.logs/',
                learning_rate=3e-4,
                gamma=0.99,
                max_grad_norm=0.5,  # Add gradient clipping
                policy_kwargs={
                    'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
                    'ortho_init': True  # Better weight initialization
                },
                device='auto'
            )
        else:
            self.model = PPO.load(self.path, env=self.env, device='auto', tensorboard_log='./.logs/',)
        
    
    def train(self):
        self._diagnose_issue()
        
        try:
            time_callback = TimeEstimateCallback(self.total_timesteps)
            checkpoint_callback = CheckpointCallback(
                save_freq=100000,
                save_path='./.checkpoints/',
                name_prefix='model'
            )
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=[time_callback, checkpoint_callback],
                tb_log_name='ppo',
                reset_num_timesteps=False
            )
            self.model.save(self.path)
        except ValueError as e:
            print(f"Training failed: {str(e)}")

    def _diagnose_issue(self):
        # Check data
        for symbol in self.data_manager.symbols:
            print(f"\nData check for {symbol}:")
            print("NaN values:", self.data_manager.data[symbol].isna().sum())
            print("Min values:", self.data_manager.data[symbol].min())
            print("Max values:", self.data_manager.data[symbol].max())
        
        # Test environment
        test_env = TradingEnvironment(self.data_manager)
        obs = test_env.reset()
        for _ in range(10):
            action = test_env.action_space.sample()
            obs, reward, done, _, _ = test_env.step(action)
            print("Reward:", reward)
            print("Observation stats - Mean:", obs.mean(), "Std:", obs.std())
        

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()