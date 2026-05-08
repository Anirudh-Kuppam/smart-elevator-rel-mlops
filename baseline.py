import random

from sim.elevator_env import ElevatorEnv

env = ElevatorEnv()

episodes = 100

total_rewards = []
total_waits = []

for episode in range(episodes):

    state = env.reset()

    done = False

    episode_reward = 0

    while not done:

        # RANDOM BASELINE
        action = random.randint(0, 3)

        next_state, reward, done = env.step(action)

        episode_reward += reward

        state = next_state

    total_rewards.append(episode_reward)
    total_waits.append(env.total_wait_time)

avg_reward = sum(total_rewards) / episodes
avg_wait = sum(total_waits) / episodes

print("\n===== BASELINE RESULTS =====")

print(f"Average Reward: {avg_reward}")
print(f"Average Wait Time: {avg_wait}")