from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from environment.envs import ReversiEnv

# 訓練和測試函數


def train_and_test_sac():
    # 創建並檢查環境
    env = ReversiEnv(render_mode="human")
    check_env(env)

    # 創建SAC模型
    model = SAC("MlpPolicy", env, verbose=1)

    # 訓練模型
    model.learn(total_timesteps=100000, log_interval=100)

    # 保存訓練好的模型
    model.save("reversi_sac_model")

    # 測試訓練好的模型
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    train_and_test_sac()
