import numpy as np
from matplotlib import pyplot as plt


class TrainingLogger:
    def __init__(self,
                 type_of_training: str,
                 ):
        self.type_of_training = type_of_training
        self.timesteps = []

        self.mean_rewards = []
        self.std_rewards = []
        self.median_rewards = []

        self.mean_ep_lengths = []
        self.std_ep_lengths = []
        self.median_ep_lengths = []

    def log(self,
            timestep,
            episode_rewards,
            episode_lengths):
        mean_reward, std_reward, median_reward = np.mean(episode_rewards), np.std(episode_rewards), np.median(
            episode_rewards)
        mean_ep_length, std_ep_length, median_ep_length = np.mean(episode_lengths), np.std(episode_lengths), np.median(
            episode_lengths)
        self.timesteps.append(timestep)
        self.mean_rewards.append(mean_reward)
        self.std_rewards.append(std_reward)
        self.median_rewards.append(median_reward)
        self.mean_ep_lengths.append(mean_ep_length)
        self.std_ep_lengths.append(std_ep_length)
        self.median_ep_lengths.append(median_ep_length)
        print(f"Episode lengths: {episode_lengths}")
        print(f"Episode rewards: {episode_rewards}")

    def plot_graph(self, x_values, y_values, std_values, x_label, y_label, title, label, filename, show_std, path):
        plt.figure(figsize=(10, 6))

        plt.plot(x_values, y_values, label=label, linewidth=2)

        if show_std:
            plt.fill_between(x_values, y_values - std_values, y_values + std_values, alpha=0.3, label='±1 Std Dev')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)

        plt.legend()
        plt.savefig(path + "/" + filename)

    def plot(self, path):
        self.plot_graph(x_values=np.array(self.timesteps),
                        y_values=np.array(self.mean_rewards),
                        std_values=np.array(self.std_rewards),
                        x_label="Timesteps",
                        y_label="Mean Reward",
                        title=f"{self.type_of_training} - Training Progress",
                        label="Mean Reward",
                        filename="mean_reward_plot.png",
                        show_std=True,
                        path=path
                        )

        self.plot_graph(x_values=np.array(self.timesteps),
                        y_values=np.array(self.mean_ep_lengths),
                        std_values=np.array(self.std_ep_lengths),
                        x_label="Timesteps",
                        y_label="Mean Episode Length",
                        title=f"{self.type_of_training} - Training Progress",
                        label="Mean Episode Length",
                        filename="mean_length_plot.png",
                        show_std=True,
                        path=path
                        )

        self.plot_graph(x_values=np.array(self.timesteps),
                        y_values=np.array(self.median_rewards),
                        std_values=[],
                        x_label="Timesteps",
                        y_label="Median Reward",
                        title=f"{self.type_of_training} - Training Progress",
                        label="Median Reward",
                        filename="median_reward_plot.png",
                        show_std=False,
                        path=path
                        )

        self.plot_graph(x_values=np.array(self.timesteps),
                        y_values=np.array(self.median_ep_lengths),
                        std_values=[],
                        x_label="Timesteps",
                        y_label="Median Episode Length",
                        title=f"{self.type_of_training} - Training Progress",
                        label="Median Episode Length",
                        filename="median_length_plot.png",
                        show_std=False,
                        path=path
                        )
