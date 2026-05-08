import random
import pickle
import csv

from sim.elevator_env import ElevatorEnv

# Create environment
env = ElevatorEnv()

# Q-table
q_table = {}

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 2000

# Possible actions
actions = [0, 1, 2, 3]

# Results storage
results = []

def get_q_value(state, action):
    return q_table.get((state, action), 0.0)

for episode in range(episodes):

    state = env.reset()

    done = False
    total_reward = 0

    while not done:

        # Epsilon-greedy exploration
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            q_values = [get_q_value(state, a) for a in actions]
            max_q = max(q_values)

            best_actions = [
                actions[i]
                for i in range(len(actions))
                if q_values[i] == max_q
            ]

            action = random.choice(best_actions)

        # Take action
        next_state, reward, done = env.step(action)

        # Current Q-value
        old_q = get_q_value(state, action)

        # Future Q-values
        future_qs = [get_q_value(next_state, a) for a in actions]
        max_future_q = max(future_qs)

        # Q-learning formula
        new_q = old_q + alpha * (
            reward + gamma * max_future_q - old_q
        )

        # Update Q-table
        q_table[(state, action)] = new_q

        # Move to next state
        state = next_state

        total_reward += reward

    # Store results
    results.append([
    episode,
    total_reward,
    env.total_wait_time,
    epsilon,
    alpha,
    gamma
    ])

    print(
        f"Episode: {episode} | "
        f"Reward: {total_reward} | "
        f"Wait Time: {env.total_wait_time}"
    )

# Save trained policy
with open("policies/policy_v1.pkl", "wb") as f:
    pickle.dump(q_table, f)

# Save experiment results
with open("results/results.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
    "episode",
    "total_reward",
    "wait_time",
    "epsilon",
    "alpha",
    "gamma"
    ])

    writer.writerows(results)

print("\nTraining Complete")
print("Policy saved in policies/")
print("Results saved in results/")