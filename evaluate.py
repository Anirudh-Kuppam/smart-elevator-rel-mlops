
from sim.elevator_env import ElevatorEnv
import pickle
import os
from dotenv import load_dotenv
import mlflow
load_dotenv()
#mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load trained policy
with open("policies/policy_v1.pkl", "rb") as f:
    q_table = pickle.load(f)

env = ElevatorEnv()

actions = [0, 1, 2, 3]


def get_q_value(state, action):
    return q_table.get((state, action), 0.0)


mlflow.set_experiment("smart-elevator-rl")

with mlflow.start_run(run_name="policy-evaluation"):

    mlflow.log_params({
        "policy_file": "policy_v1.pkl",
        "num_actions": len(actions),
        "mode": "greedy",
    })

    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("\n===== TRAINED ELEVATOR AGENT =====")

    while not done:

        env.render()

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

        # Log each step
        mlflow.log_metrics({
            "step_reward": reward,
            "chosen_action": action,
        }, step=steps)

        total_reward += reward
        state = next_state
        steps += 1

    mlflow.log_metrics({
        "final_reward": total_reward,
        "total_wait_time": env.total_wait_time,
        "total_steps": steps,
    })

    print("\n===== EVALUATION COMPLETE =====")
    print(f"Final Reward: {total_reward}")
    print(f"Total Wait Time: {env.total_wait_time}")
