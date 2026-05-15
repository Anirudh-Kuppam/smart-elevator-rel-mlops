
import random
from dotenv import load_dotenv
import mlflow
load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

from sim.elevator_env import ElevatorEnv

env = ElevatorEnv()

episodes = 100
total_rewards = []
total_waits = []

mlflow.set_experiment("smart-elevator-rl")

with mlflow.start_run(run_name="random-baseline"):

    mlflow.log_params({
        "episodes": episodes,
        "policy": "random",
        "num_actions": 4,
    })

    for episode in range(episodes):

        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = random.randint(0, 3)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)
        total_waits.append(env.total_wait_time)

        mlflow.log_metrics({
            "episode_reward": episode_reward,
            "wait_time": env.total_wait_time,
        }, step=episode)

    avg_reward = sum(total_rewards) / episodes
    avg_wait = sum(total_waits) / episodes

    mlflow.log_metrics({
        "avg_reward": avg_reward,
        "avg_wait_time": avg_wait,
    })

    print("\n===== BASELINE RESULTS =====")
    print(f"Average Reward: {avg_reward}")
    print(f"Average Wait Time: {avg_wait}")
