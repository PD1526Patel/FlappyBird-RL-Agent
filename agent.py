import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.network_sync_rate = params["network_sync_rate"]
        self.reward_thresold = params["reward_thresold"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def run(self, is_training=True, render=False):
        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
        )

        num_states = env.observation_space.shape[0]  # input dimensions
        num_actions = env.action_space.n  # output dimensions

        policy_dqn = DQN(
            num_states,
            num_actions,
        ).to(device)

        if is_training:
            memory = ReplayMemory(maxlen=self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device=device)
            # Copy the weight and bias values from policy network to target network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0

            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
        else:
            # Load Best Model,  i.e. best policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            epsilon = None

        try:

            for episode in itertools.count():
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float, device=device)
                episode_reward = 0
                terminated = False

                while not terminated and episode_reward < self.reward_thresold:
                    # Next action:
                    # (feed the observation to your agent here)

                    if is_training and random.random() < epsilon:
                        action = env.action_space.sample()  # Exploration
                        action = torch.tensor(action, dtype=torch.long, device=device)
                    else:
                        with torch.no_grad():
                            action = (
                                policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                            )  # Exploitation

                    # Processing: Terminated => Done

                    next_state, reward, terminated, _, _ = env.step(action.item())
                    episode_reward += reward

                    # Create Tensors
                    reward = torch.tensor(reward, dtype=torch.float, device=device)
                    next_state = torch.tensor(
                        next_state, dtype=torch.float, device=device
                    )

                    if is_training:
                        memory.append((state, action, next_state, reward, terminated))
                        steps += 1
                        print(
                            f"Episodes = {episode+1} & Total Reward = {episode_reward} & Epsilon = {epsilon}"
                        )

                    # Checking if the player is still alive
                    if terminated:
                        break

                    state = next_state

                if is_training:
                    # Epsilon Decay
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                    if episode_reward > best_reward:
                        log_msg = (
                            f"best reward = {episode_reward} for episode = {episode+1}"
                        )

                        with open(self.LOG_FILE, "a") as f:
                            f.write(log_msg + "\n")

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        best_reward = episode_reward

                if is_training and len(memory) > self.mini_batch_size:
                    # get sample
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Sync the networks
                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0
        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C), Closing environment....")
        finally:
            env.close()
        # env.close()
        # mannually stop

    def optimize_slow(self, mini_batch, policy_dqn, target_dqn):
        # Get Experience => Train in batch to fast training, rather trainig one by one sample

        for state, action, next_state, reward, terminated in mini_batch:

            if terminated:
                target = reward

            else:
                with torch.no_grad():
                    # y_true = target_q
                    target_q = reward + self.gamma * target_dqn(next_state).max()

            # y_predicted = current_q
            current_q = policy_dqn(state)

            # Loss
            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Get Experience =>
        # Train in batch to fast training,
        # rather trainig one by one sample

        # get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device=device)

        # Calculate target Q-values
        # if terminations = true => zero

        with torch.no_grad():
            target_q = (
                rewards
                + (1 - terminations)
                * self.gamma
                * target_dqn(next_states).max(dim=1)[0]
            )

        # Calculate y_pred i.e. Q-Value from current policy
        current_q = (
            policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description="Train or Test model.")
    parser.add_argument("hyperparameters", help="")
    parser.add_argument("--train", help="Training Mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
