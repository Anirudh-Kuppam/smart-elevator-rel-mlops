import pickle

from sim.elevator_env import ElevatorEnv

# Load RL policy
with open("policies/policy_v1.pkl", "rb") as f:
    q_table = pickle.load(f)

env = ElevatorEnv()

actions = [0, 1, 2, 3]

episodes = 100

total_rewards = []
total_waits = []

def get_q_value(state, action):
    return q_table.get((state, action), 0.0)

for episode in range(episodes):

    state = env.reset()

    done = False

    episode_reward = 0

    while not done:

        q_values = [get_q_value(state, a) for a in actions]

        max_q = max(q_values)

        best_actions = [
            actions[i]
            for i in range(len(actions))
            if q_values[i] == max_q
        ]

        action = best_actions[0]

        next_state, reward, done = env.step(action)

        episode_reward += reward

        state = next_state

    total_rewards.append(episode_reward)
    total_waits.append(env.total_wait_time)

avg_reward = sum(total_rewards) / episodes
avg_wait = sum(total_waits) / episodes

print("\n===== RL RESULTS =====")

print(f"Average Reward: {avg_reward}")
print(f"Average Wait Time: {avg_wait}")