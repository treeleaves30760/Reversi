import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment.envs import ReversiEnv


def train_and_test_ppo():
    # 創建並檢查環境
    env = ReversiEnv(render_mode="human")
    check_env(env)

    # 創建PPO模型
    model = PPO("MlpPolicy", env, verbose=1)

    # 訓練模型
    model.learn(total_timesteps=1000000, log_interval=100, progress_bar=True)

    # 保存訓練好的模型
    model.save(f"models/ppo_{time.time()}")

    # 測試訓練好的模型
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    train_and_test_ppo()
