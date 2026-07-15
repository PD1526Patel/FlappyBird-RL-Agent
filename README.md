# Flappy Bird DQN (Reinforcement Learning) Agent

A Deep Q-Network (DQN) Reinforcement Learning agent built in PyTorch to play Flappy Bird using the `flappy-bird-gymnasium` environment.

## Agent Demo

A recording of the trained agent playing Flappy Bird.

https://github.com/user-attachments/assets/video.mp4
_Click the GIF to open the full-resolution video._

---

## Project Overview

This repository implements a Deep Q-Learning agent that learns to navigate the obstacles in Flappy Bird:

- **Environment**: [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) (based on OpenAI Gymnasium).
- **DQN Architecture** (`dqn.py`): A Multi-Layer Perceptron (MLP) with a hidden dimension of 256, mapping 12 state variables (distances to obstacles, velocities, etc.) to 2 actions (flap or glide).
- **Experience Replay** (`experience_replay.py`): A memory buffer of size 100,000 to store transitions and sample batches uniformly, stabilizing the Q-learning updates.
- **Stable Training** (`agent.py`): Implements a separate Policy DQN and Target DQN, with periodic weight synchronization (every 10 steps) to prevent oscillation.

---

## Hyperparameter Settings (`parameters.yaml`)

| Hyperparameter                       | Value     | Description                                            |
| ------------------------------------ | --------- | ------------------------------------------------------ |
| **Epsilon Initial (`epsilon_init`)** | `1.0`     | Initial exploration rate                               |
| **Epsilon Min (`epsilon_min`)**      | `0.05`    | Minimum exploration rate                               |
| **Epsilon Decay (`epsilon_decay`)**  | `0.995`   | Factor to decay epsilon per episode                    |
| **Replay Memory Size**               | `100,000` | Size of experience replay buffer                       |
| **Mini Batch Size**                  | `32`      | Number of experiences sampled for training             |
| **Network Sync Rate**                | `10`      | Steps between copying policy network to target network |
| **Learning Rate ($\alpha$)**         | `0.001`   | Adam Optimizer learning rate                           |
| **Discount Factor ($\gamma$)**       | `0.99`    | Future reward discount factor                          |
| **Reward Threshold**                 | `1000`    | Game score target threshold                            |

---

## Setup and Usage

### Prerequisites

Make sure you have python 3.8+ and install the dependencies:

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml pygame
```

### Running the Project

#### 1. Train the Agent

To train a new DQN model on the default parameter set:

```bash
python agent.py flappybirdv0 --train
```

#### 2. Test / Evaluate the Agent

To load the best-saved model (weights saved in `runs/flappybirdv0.pt`) and render the game window:

```bash
python agent.py flappybirdv0
```

#### 3. Play Manually

If you want to play the game yourself using the `SPACE` key to flap:

```bash
python game_flappy_bird.py
```
