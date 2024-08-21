import time
import argparse

import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.envs import ReversiEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test Reversi AI")
    parser.add_argument("--model", type=str, default="PPO", choices=["PPO", "A2C", "DQN"],
                        help="RL algorithm to use (default: PPO)")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total timesteps for training (default: 1000000)")
    parser.add_argument("--learning-rate", type=float, default=0.0003,
                        help="Learning rate (default: 0.0003)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training (default: 64)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps to run for each environment per update (default: 2048)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--test-episodes", type=int, default=10,
                        help="Number of episodes to test after training (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during testing")
    return parser.parse_args()


def create_model(args, env):
    if args.model == "PPO":
        return PPO("MlpPolicy", env, learning_rate=args.learning_rate,
                   n_steps=args.n_steps, batch_size=args.batch_size,
                   gamma=args.gamma, verbose=1)
    elif args.model == "A2C":
        return A2C("MlpPolicy", env, learning_rate=args.learning_rate,
                   n_steps=args.n_steps, gamma=args.gamma, verbose=1)
    elif args.model == "DQN":
        return DQN("MlpPolicy", env, learning_rate=args.learning_rate,
                   batch_size=args.batch_size, gamma=args.gamma, verbose=1)


def train_and_test_model():
    args = parse_args()

    # Create and check environment
    env = ReversiEnv(render_mode="human" if args.render else None)
    check_env(env)
    env = DummyVecEnv([lambda: env])

    # Create model
    model = create_model(args, env)

    # Train model
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Save trained model
    model_filename = f"models/{args.model}_{int(time.time())}"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")

    # Test trained model
    obs = env.reset()
    for episode in range(args.test_episodes):
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            if args.render:
                env.render()
        print(f"Episode {episode + 1} reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    train_and_test_model()
