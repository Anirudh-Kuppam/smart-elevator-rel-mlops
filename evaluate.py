import pickle

from sim.elevator_env import ElevatorEnv

# Load trained policy
with open("policies/policy_v1.pkl", "rb") as f:
    q_table = pickle.load(f)

env = ElevatorEnv()

actions = [0, 1, 2, 3]

def get_q_value(state, action):
    return q_table.get((state, action), 0.0)

state = env.reset()

done = False
total_reward = 0

print("\n===== TRAINED ELEVATOR AGENT =====")

while not done:

    env.render()

    # Choose best action
    q_values = [get_q_value(state, a) for a in actions]

    max_q = max(q_values)

    best_actions = [
        actions[i]
        for i in range(len(actions))
        if q_values[i] == max_q
    ]

    action = best_actions[0]

    print(f"Chosen Action: {action}")

    next_state, reward, done = env.step(action)

    print(f"Reward: {reward}")

    total_reward += reward

    state = next_state

print("\n===== EVALUATION COMPLETE =====")

print(f"Final Reward: {total_reward}")
print(f"Total Wait Time: {env.total_wait_time}")