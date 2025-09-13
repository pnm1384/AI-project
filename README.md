# AI-project
Phase 1: Image Segmentation with UNet, AttentionUNet, and ResidualAttentionUNet
📌 Project Overview
This project was developed as Phase 1 of the Artificial Intelligence course (Spring 2025, Sharif University of Technology, Computer Engineering Department). It focuses on implementing image segmentation using the Massachusetts Road Dataset.

We explore and build from scratch the following architectures using PyTorch:

UNet
Attention UNet
Residual Attention UNet
The models are trained with a composite loss function:

IoU Loss + Dice Loss + BCE Loss
Optimization is performed with AdamW along with a learning rate scheduler to stabilize convergence.

👨‍💻 Authors

Matin Bagheri

Ali HosseinKhalaj

Parsa NorouziManesh

📂 Dataset
We use the Massachusetts Roads Dataset, a benchmark dataset for road extraction from aerial images.

Input images: 1500 aerial RGB images of size 1500×1500 pixels
Ground truth masks: Binary masks indicating road vs. non-road pixels
Splits: Training, Validation, and Test sets
In this project:

Images are resized to 256×256 for training efficiency.
Masks are normalized and thresholded for binary segmentation.
This dataset is widely used for evaluating deep learning methods in semantic segmentation.

🏗️ Implementation Details
The project consists of three main stages:

1. Data Preparation
Loaded the dataset from Kaggle using kagglehub.
Applied preprocessing: resizing, normalization, tensor conversion.
Implemented a custom RoadDataset class for PyTorch.
2. Model Architectures
UNet: Encoder-decoder with skip connections.
Attention UNet: Adds attention gates to focus on relevant spatial features.
Residual Attention UNet: Combines residual blocks with attention gates for deeper and more efficient learning.
3. Training Setup
Optimizer: AdamW
Scheduler: Learning rate scheduler for stable convergence
Loss Function: IoU Loss + Dice Loss + BCE Loss
Metrics: Validation loss, IoU Score, Dice Score
📊 Results
The models were trained and evaluated on the Massachusetts Roads dataset.

Model	Num Epochs	Val Loss	IoU Score	Dice Score
UNet	15	0.59	0.69	0.81
Attention UNet	15	0.61	0.67	0.80
Residual Attention UNet	15	0.59	0.68	0.81
Observations
UNet achieved the highest IoU (0.69) and Dice (0.81), serving as a strong baseline.
Attention UNet slightly underperformed compared to vanilla UNet, suggesting that the attention mechanism did not yield significant benefits within the given training setup and dataset size.
Residual Attention UNet performed similarly to UNet, showing stable Dice scores but no substantial improvement in IoU or loss reduction.
👉 Overall, all three models converge to very close performance, with UNet providing the best balance under the current configuration.

📜 License
This project is for educational purposes as part of the AI course at Sharif University of Technology.

# 🤖 Phase 2: Soft Actor-Critic (SAC) for Continuous Control
## 📌 Project Overview
This project was developed as Phase 2 of the Artificial Intelligence course (Spring 2025, Sharif University of Technology, Computer Engineering Department). It focuses on implementing the Soft Actor-Critic (SAC) algorithm from scratch to solve continuous control problems in a physics-based environment.

We implemented the core components using PyTorch, including:

- **Actor Network**: A policy network for action selection.

- **Critic Networks**: Two separate Q-value networks for value estimation.

- **Replay Buffer**: An off-policy memory to stabilize training.

The agent's objective is to maximize a combination of expected return and policy entropy, encouraging robust exploration.

## 👨‍💻 Authors

Matin Bagheri

Parsa NorouziManesh

Ali HosseinKhalaj

## 📜 Core Algorithm: Soft Actor-Critic (SAC)
Soft Actor-Critic is a modern, off-policy reinforcement learning algorithm particularly well-suited for continuous action spaces. Unlike traditional algorithms that solely focus on maximizing reward, SAC introduces an entropy-regularized objective. This means the agent is trained to maximize not only the cumulative reward but also the randomness or entropy of its policy.

This dual objective serves two key purposes:

- **Enhanced Exploration**: By encouraging a more random policy, the agent is less likely to get stuck in local optima.

- **Increased Stability**: The entropy term acts as a regularization factor, leading to more stable and robust learning.

As an actor-critic method, it uses an actor network to learn the policy and a critic network to evaluate the value of states and actions, which in turn guides the actor's learning. The use of a replay buffer makes it highly sample-efficient, allowing it to learn from past experiences.

## 📂 Environment
We use the HalfCheetah-v5 environment from the Gymnasium and PyBullet libraries. This is a classic continuous control task where the agent must learn to run as fast as possible without falling over.

- **State Space**: A continuous vector representing the robot's joint angles and velocities.

- **Action Space**: A continuous vector of torques applied to the robot's joints.

- **Physics**: PyBullet provides a fast and robust physics engine for simulating the environment.

## 🏗️ Implementation Details & Network Architectures
The project was structured into three main stages.

### 1. Data Management
Replay Buffer: We implemented a ReplayBuffer to store and manage transitions of the form (state, action, reward, next_state, done). This off-policy memory allows us to sample random mini-batches for training, effectively breaking temporal correlations in the data and stabilizing the learning process.

### 2. Network Architectures
Actor Network: The actor is a feed-forward neural network that outputs the parameters (μ, σ) of a stochastic Gaussian policy. Actions are sampled from this distribution using the reparameterization trick, which allows gradients to flow back through the sampling process. A tanh activation function is applied to these actions, ensuring they are bounded within the environment's action space.

Twin Critic Networks: Two separate critic networks were implemented to estimate the soft Q-value function, Q(s,a). The use of two critics, and taking the minimum of their outputs to compute the Bellman target, is a crucial technique in SAC to mitigate overestimation bias and improve training stability.

### 3. Agent & Training Setup
Agent Class: The Agent class integrates the actor, critics, and replay buffer. It orchestrates the training loop, managing the delayed policy updates, soft target network updates, and the crucial automatic entropy tuning.

Loss Function: We used Mean Squared Error (MSE) for the critic's value loss and an entropy-regularized policy gradient for the actor's policy loss.

Optimizer: All networks were trained using the AdamW optimizer, a robust choice for deep learning.

Automatic Entropy Tuning: The agent was configured to automatically adjust the entropy regularization coefficient (α) to match a predefined target_entropy, thus dynamically balancing exploration and exploitation throughout training.

## 📊 Results
The SAC agent was trained on the HalfCheetah-v5 environment, successfully learning a robust running policy.

- **Performance**: The learning curve shows a stable and consistent increase in average scores over time, indicating that the agent successfully converged on an optimal policy.

- **Effectiveness**: The recorded video of the trained agent demonstrates a smooth, well-coordinated running motion, showcasing the effectiveness of the algorithm in learning a complex continuous control task.

👉 Overall, this project demonstrates a successful implementation of the Soft Actor-Critic algorithm, capable of solving a continuous control problem by effectively balancing reward maximization with a focus on active exploration.

## 📜 License
This project is for educational purposes as part of the AI course at Sharif University of Technology.
