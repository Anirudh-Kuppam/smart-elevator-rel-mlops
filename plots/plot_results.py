import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/results.csv")

# Reward Plot
plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["total_reward"])

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Over Episodes")

plt.grid(True)

plt.savefig("plots/reward_plot.png")

# Wait Time Plot
plt.figure(figsize=(10,5))
plt.plot(df["episode"], df["wait_time"])

plt.xlabel("Episode")
plt.ylabel("Wait Time")
plt.title("Wait Time Over Episodes")

plt.grid(True)

plt.savefig("plots/wait_time_plot.png")

print("Plots saved in plots/")