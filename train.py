from sim.elevator_env import ElevatorEnv
import random
import pickle
import csv

import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("sqlite:///mlflow.db")


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


mlflow.set_experiment("smart-elevator-rl")

with mlflow.start_run(run_name="q-learning-train"):

    # Log hyperparameters
    mlflow.log_params({
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "episodes": episodes,
        "num_actions": len(actions),
    })

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

            # Q-learning update
            old_q = get_q_value(state, action)
            future_qs = [get_q_value(next_state, a) for a in actions]
            max_future_q = max(future_qs)
            new_q = old_q + alpha * (reward + gamma * max_future_q - old_q)
            q_table[(state, action)] = new_q

            state = next_state
            total_reward += reward

        # Log per-episode metrics to MLflow
        mlflow.log_metrics({
            "episode_reward": total_reward,
            "wait_time": env.total_wait_time,
            "q_table_size": len(q_table),
        }, step=episode)

        results.append(
            [episode, total_reward, env.total_wait_time, epsilon, alpha, gamma])

        print(
            f"Episode: {episode} | "
            f"Reward: {total_reward} | "
            f"Wait Time: {env.total_wait_time}"
        )

    # Aggregate summary metrics
    all_rewards = [r[1] for r in results]
    all_waits = [r[2] for r in results]
    mlflow.log_metrics({
        "avg_reward": sum(all_rewards) / episodes,
        "avg_wait_time": sum(all_waits) / episodes,
        "best_episode_reward": max(all_rewards),
        "final_q_table_size": len(q_table),
    })

    # Save and log policy artifact
    policy_path = "policies/policy_v1.pkl"
    with open(policy_path, "wb") as f:
        pickle.dump(q_table, f)
    mlflow.log_artifact(policy_path, artifact_path="policy")

    # Save and log results CSV
    csv_path = "results/results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward",
                        "wait_time", "epsilon", "alpha", "gamma"])
        writer.writerows(results)
    mlflow.log_artifact(csv_path, artifact_path="results")

    print("\nTraining Complete")
    print("Policy saved in policies/")
    print("Results saved in results/")
    print("MLflow run complete — view at http://localhost:5000")
