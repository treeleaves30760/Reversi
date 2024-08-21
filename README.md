# Reversi

This is a project that provide Reversi Environment

## Installation

You can create a virtual environment for this project.

## Usage

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | string | "PPO" | Choose the reinforcement learning algorithm. Options: "PPO", "A2C", "DQN". |
| `--timesteps` | integer | 1000000 | Total number of timesteps to train the model. Higher values generally lead to better performance but longer training times. |
| `--learning-rate` | float | 0.0003 | Learning rate for the optimizer. Controls how quickly the model adapts to the problem. |
| `--batch-size` | integer | 64 | Number of samples used in each training iteration. Larger batch sizes can lead to more stable training but require more memory. |
| `--n-steps` | integer | 2048 | Number of steps to run for each environment per update (for PPO and A2C). Affects the trade-off between sample efficiency and computational efficiency. |
| `--gamma` | float | 0.99 | Discount factor for future rewards. Values closer to 1 make the agent more far-sighted. |
| `--test-episodes` | integer | 10 | Number of episodes to run for testing the trained model. More episodes provide a better estimate of the model's performance. |
| `--render` | flag | False | If set, renders the environment during testing. Useful for visualizing the agent's behavior. |

To use these arguments, append them to your command when running the script. For example:

```bash
python train_reversi.py --model A2C --timesteps 500000 --learning-rate 0.0001 --test-episodes 20
```

This command would train an A2C model for 500,000 timesteps with a learning rate of 0.0001 and test it for 20 episodes.
