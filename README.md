# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of states:", n_states)
print("Number of actions:", n_actions)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
gamma = 0.99
alpha = 0.1
max_steps_per_episode = 100

# Fixed epsilon
fixed_epsilon = 0.9

# Variable / decaying epsilon
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# Generate One Complete Episode
# -------------------------------------------------

def generate_episode(Q, epsilon):
    episode = []

    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(Q, state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))
        state = next_state

        if terminated or truncated:
            break

    return episode

# -------------------------------------------------
# Monte Carlo Control
# -------------------------------------------------

def monte_carlo_control(epsilon_mode):
    Q = np.zeros((n_states, n_actions))
    episode_rewards = []

    if epsilon_mode == "fixed":
        epsilon = fixed_epsilon
    elif epsilon_mode == "variable":
        epsilon = epsilon_start
    else:
        raise ValueError("epsilon_mode must be 'fixed' or 'variable'")

    for episode_idx in range(num_episodes):

        # Generate complete episode
        episode = generate_episode(Q, epsilon)

        # Track total reward
        total_reward = sum(reward for _, _, reward in episode)
        episode_rewards.append(total_reward)

        # First-visit Monte Carlo update
        visited = set()
        G = 0.0

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]

            G = reward + gamma * G

            if (state, action) not in visited:
                visited.add((state, action))

                # Constant step-size Monte Carlo update
                Q[state, action] += alpha * (G - Q[state, action])

        # Decay epsilon only for variable-epsilon experiment
        if epsilon_mode == "variable":
            epsilon = max(
                epsilon_min,
                epsilon * epsilon_decay
            )

    return Q, episode_rewards

# -------------------------------------------------
# Run Fixed Epsilon Experiment
# -------------------------------------------------

print("Running Fixed Epsilon Monte Carlo Control...")

Q_fixed, rewards_fixed = monte_carlo_control("fixed")

print("Fixed epsilon experiment completed.")
print("Final epsilon:", fixed_epsilon)

# -------------------------------------------------
# Run Variable Epsilon Experiment
# -------------------------------------------------

print("Running Variable Epsilon Monte Carlo Control...")

Q_variable, rewards_variable = monte_carlo_control("variable")

print("Variable epsilon experiment completed.")
print("Minimum epsilon:", epsilon_min)

# -------------------------------------------------
# Extract Greedy Policies
# -------------------------------------------------

optimal_policy_fixed = np.argmax(Q_fixed, axis=1)
state_values_fixed = np.max(Q_fixed, axis=1)

optimal_policy_variable = np.argmax(Q_variable, axis=1)
state_values_variable = np.max(Q_variable, axis=1)

# -------------------------------------------------
# Policy Display
# -------------------------------------------------

def print_policy(policy, title):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\n" + title)
    print(policy_grid)


def print_value_function(values, title):
    print("\n" + title)
    print(np.round(values.reshape(4, 4), 3))


print_policy(
    optimal_policy_fixed,
    "Learned Policy - Fixed Epsilon"
)

print_value_function(
    state_values_fixed,
    "Estimated State-Value Function - Fixed Epsilon"
)

print_policy(
    optimal_policy_variable,
    "Learned Policy - Variable Epsilon"
)

print_value_function(
    state_values_variable,
    "Estimated State-Value Function - Variable Epsilon"
)

# -------------------------------------------------
# Final Q-Tables
# -------------------------------------------------

print("\nFinal Q-table - Fixed Epsilon:")
print(np.round(Q_fixed, 3))

print("\nFinal Q-table - Variable Epsilon:")
print(np.round(Q_variable, 3))

# -------------------------------------------------
# Final Performance
# -------------------------------------------------

last_n = 1000

average_reward_fixed = np.mean(rewards_fixed[-last_n:])
average_reward_variable = np.mean(rewards_variable[-last_n:])

success_rate_fixed = np.mean(
    np.array(rewards_fixed[-last_n:]) > 0
) * 100

success_rate_variable = np.mean(
    np.array(rewards_variable[-last_n:]) > 0
) * 100

print("==============================================")
print("             FINAL RESULTS")
print("==============================================")

print(f"Fixed Epsilon ({fixed_epsilon}):")
print(f"  Average reward (last {last_n}): {average_reward_fixed:.4f}")
print(f"  Success rate (last {last_n}): {success_rate_fixed:.2f}%")

print("\nVariable Epsilon:")
print(f"  Epsilon range: {epsilon_start} -> {epsilon_min}")
print(f"  Average reward (last {last_n}): {average_reward_variable:.4f}")
print(f"  Success rate (last {last_n}): {success_rate_variable:.2f}%")

# -------------------------------------------------
# Moving Average Learning Curves
# -------------------------------------------------

window = 500

moving_fixed = np.convolve(
    rewards_fixed,
    np.ones(window) / window,
    mode="valid"
)

moving_variable = np.convolve(
    rewards_variable,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(9, 5))

plt.plot(
    moving_fixed,
    label=f"Fixed Epsilon (ε = {fixed_epsilon})"
)

plt.plot(
    moving_variable,
    label=f"Variable Epsilon ({epsilon_start} → {epsilon_min})"
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Fixed Epsilon vs Variable Epsilon")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------
# Direct Numerical Comparison
# -------------------------------------------------

print("\n========== COMPARISON ==========")

print(f"Fixed Epsilon Average Reward:    {average_reward_fixed:.4f}")
print(f"Variable Epsilon Average Reward: {average_reward_variable:.4f}")

print(f"Fixed Epsilon Success Rate:      {success_rate_fixed:.2f}%")
print(f"Variable Epsilon Success Rate:   {success_rate_variable:.2f}%")

# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()

```

---

## Output

```text
Final Q-table - Fixed Epsilon:
[[0.01  0.018 0.008 0.023]
 [0.022 0.    0.044 0.043]
 [0.029 0.114 0.05  0.055]
 [0.078 0.    0.002 0.001]
 [0.031 0.03  0.    0.066]
 [0.    0.    0.    0.   ]
 [0.    0.408 0.    0.073]
 [0.    0.    0.    0.   ]
 [0.052 0.    0.123 0.139]
 [0.088 0.276 0.304 0.   ]
 [0.247 0.598 0.    0.161]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.214 0.61  0.188]
 [0.248 0.656 1.    0.124]
 [0.    0.    0.    0.   ]]

Final Q-table - Variable Epsilon:
[[0.737 0.916 0.654 0.603]
 [0.685 0.    0.608 0.604]
 [0.701 0.659 0.584 0.703]
 [0.59  0.    0.222 0.298]
 [0.727 0.926 0.    0.731]
 [0.    0.    0.    0.   ]
 [0.    0.821 0.    0.712]
 [0.    0.    0.    0.   ]
 [0.757 0.    0.97  0.818]
 [0.883 0.886 0.98  0.   ]
 [0.893 0.99  0.    0.9  ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.923 0.99  0.954]
 [0.937 0.99  1.    0.979]
 [0.    0.    0.    0.   ]]



Learned Policy - Fixed Epsilon
[['U' 'R' 'D' 'L']
 ['U' 'L' 'D' 'L']
 ['U' 'R' 'D' 'L']
 ['L' 'R' 'R' 'L']]

Estimated State-Value Function - Fixed Epsilon
[[0.023 0.044 0.114 0.078]
 [0.066 0.    0.408 0.   ]
 [0.139 0.304 0.598 0.   ]
 [0.    0.61  1.    0.   ]]

Learned Policy - Variable Epsilon
[['D' 'L' 'U' 'L']
 ['D' 'L' 'D' 'L']
 ['R' 'R' 'D' 'L']
 ['L' 'R' 'R' 'L']]

Estimated State-Value Function - Variable Epsilon
[[0.916 0.685 0.703 0.59 ]
 [0.926 0.    0.821 0.   ]
 [0.97  0.98  0.99  0.   ]
 [0.    0.99  1.    0.   ]]



Average reward over last 1000 episodes: 
Fixed Epsilon (0.9):
  Average reward (last 1000): 0.0330
  Success rate (last 1000): 3.30%

Variable Epsilon:
  Epsilon range: 1.0 -> 0.05
  Average reward (last 1000): 0.9380
  Success rate (last 1000): 93.80%
```


---

## Result
```text



```
---

## Inference
```text



```





---

