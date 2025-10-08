# RL_grind 🚀

A comprehensive reinforcement learning implementation repository covering fundamental algorithms from basic concepts to deep RL methods. This repository showcases practical implementations of various RL algorithms with real environment testing and visualization.

## 📚 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Visualizations](#results--visualizations)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## 🎯 Overview

This repository contains implementations of reinforcement learning algorithms organized from basic concepts to advanced techniques:

- **Basics of RL**: Environment exploration and gymnasium fundamentals
- **Classical RL**: Multi-armed bandits, Monte Carlo methods, and Temporal Difference learning
- **Deep Q-Learning**: Neural network-based Q-learning for complex environments
- **Policy Gradients**: REINFORCE algorithm with and without baseline

All implementations include practical examples using popular environments like LunarLander and Pong.

## 📁 Repository Structure

```
RL_grind/
├── Basics_of_RL/                    # Fundamental RL concepts
│   ├── example1.py                  # Basic gymnasium environment setup
│   ├── example2.py                  # Environment interaction examples
│   └── lander.gif                   # LunarLander visualization
├── Classical_RL_algos/              # Traditional RL algorithms
│   ├── egreedy/                     # Epsilon-greedy multi-armed bandit
│   │   ├── egreedy.py              # Bandit implementation
│   │   └── *.svg                   # Visualization plots
│   ├── monte_carlo/                 # Monte Carlo methods
│   │   ├── monte_carlo.py          # MC implementation for LunarLander
│   │   └── lunarlander_videos_mc/  # Training videos
│   └── temporaldifference/          # TD learning algorithms
│       ├── temporaldifference.py   # Q-learning implementation
│       └── lunarlander_videos_td/  # Training videos
├── Deep_Q_Learning/                 # Neural network-based Q-learning
│   ├── dqn_pong.py                 # DQN for Pong environment
│   ├── dqn_play.py                 # Trained model playback
│   ├── lib/                        # DQN utilities and wrappers
│   └── ALE/                        # Saved model checkpoints
└── Policy_Gradients/                # Policy-based methods
    ├── cartpole_reinforce.py       # Basic REINFORCE algorithm
    └── cartpole_reinforce_baseline.py # REINFORCE with baseline
```

## 🧠 Algorithms Implemented

### 1. Multi-Armed Bandits
- **Epsilon-Greedy Strategy**: Balances exploration vs exploitation
- **Features**: Reward distribution visualization, convergence analysis
- **Environment**: Custom bandit environment

### 2. Monte Carlo Methods
- **First-Visit Monte Carlo**: Learning from complete episodes
- **Environment**: LunarLander-v3
- **Features**: State discretization, episode recording, TensorBoard logging

### 3. Temporal Difference Learning
- **Q-Learning (TD(0))**: Model-free off-policy learning
- **Environment**: LunarLander-v3
- **Optimizations**: Vectorized operations, efficient state discretization

### 4. Deep Q-Network (DQN)
- **Neural Q-Learning**: Deep neural networks for Q-function approximation
- **Environment**: Atari Pong
- **Features**: Experience replay, target network, frame preprocessing

### 5. Policy Gradient Methods
- **REINFORCE**: Direct policy optimization
- **REINFORCE with Baseline**: Variance reduction using value function baseline
- **Environment**: CartPole-v1

## ⚙️ Installation

### Prerequisites
```bash
# Python 3.8+ required
python --version
```

### Dependencies

#### Core Dependencies
```bash
pip install gymnasium numpy matplotlib seaborn torch tensorboard
```

#### For Atari Games (DQN)
```bash
pip install "gymnasium[atari,accept-rom-license]==0.29.1" "ale-py==0.8.1"
```

#### For LunarLander
```bash
pip install gymnasium[box2d]
```

### Clone Repository
```bash
git clone https://github.com/Venkat-Shadeslayer/RL_grind.git
cd RL_grind
```

## 🚀 Usage

### 1. Explore RL Basics
```bash
cd Basics_of_RL
python example1.py  # Basic environment setup
python example2.py  # Environment interactions
```

### 2. Multi-Armed Bandits
```bash
cd Classical_RL_algos/egreedy
python egreedy.py  # Run epsilon-greedy bandit experiment
```

### 3. Monte Carlo Learning
```bash
cd Classical_RL_algos/monte_carlo
python monte_carlo.py  # Train LunarLander with Monte Carlo
```

### 4. Q-Learning (Temporal Difference)
```bash
cd Classical_RL_algos/temporaldifference
python temporaldifference.py  # Train LunarLander with Q-learning
```

### 5. Deep Q-Network
```bash
cd Deep_Q_Learning
python dqn_pong.py  # Train DQN on Pong
python dqn_play.py  # Play with trained model
```

### 6. Policy Gradients
```bash
cd Policy_Gradients
python cartpole_reinforce.py           # Basic REINFORCE
python cartpole_reinforce_baseline.py  # REINFORCE with baseline
```

## 📊 Results & Visualizations

### TensorBoard Monitoring
Most algorithms include TensorBoard logging for training visualization:
```bash
tensorboard --logdir runs/
```

### Available Visualizations
- **Bandit**: Reward distributions, action selection analysis
- **Monte Carlo/Q-Learning**: Training curves, episode rewards, video recordings
- **DQN**: Loss curves, reward progression, game performance
- **Policy Gradients**: Episode rewards, policy loss tracking

### Saved Content
- Training videos for successful episodes
- Model checkpoints for trained agents
- Performance plots and analysis graphs

## 🔧 Key Features

- **Modular Design**: Each algorithm is self-contained and well-documented
- **Visualization**: Comprehensive plotting and video recording
- **Logging**: TensorBoard integration for training monitoring
- **Optimization**: Efficient implementations with vectorized operations
- **Flexibility**: Easy parameter tuning and experimentation

## 📋 Dependencies Summary

```txt
gymnasium[atari,box2d]>=0.29.1
ale-py==0.8.1
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
tensorboard>=2.7.0
```

## 🤝 Contributing

Feel free to contribute by:
- Adding new RL algorithms
- Improving existing implementations
- Adding more environments
- Enhancing documentation
- Reporting bugs or issues

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🎯 Future Work

- [ ] Actor-Critic methods
- [ ] Advanced DQN variants (Double DQN, Dueling DQN)
- [ ] Proximal Policy Optimization (PPO)
- [ ] Multi-agent reinforcement learning
- [ ] Continuous control algorithms

---

**Happy Learning! 🎉**

*This repository represents a comprehensive journey through reinforcement learning, from theoretical foundations to practical implementations.*
