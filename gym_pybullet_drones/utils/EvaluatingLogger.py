import numpy as np
from matplotlib import pyplot as plt


class EvaluatingLogger:
    def __init__(self):
        self.rewards = []
        self.timesteps = []
        self.counter = 0

    def log(self, reward):
        self.rewards.append(reward)
        self.timesteps.append(self.counter)
        self.counter += 1

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

    def plot_two_graphs_side_by_side(
            self,
            x1, y1, std1, x1_label, y1_label, title1, label1,
            x2, y2, std2, x2_label, y2_label, title2, label2,
            show_std,
            filename,
            path
    ):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # First graph
        axes[0].plot(x1, y1, label=label1, linewidth=2)
        if show_std:
            axes[0].fill_between(x1, y1 - std1, y1 + std1, alpha=0.3, label='±1 Std Dev')
        axes[0].set_xlabel(x1_label)
        axes[0].set_ylabel(y1_label)
        axes[0].set_title(title1)
        axes[0].legend()

        # Second graph
        axes[1].plot(x2, y2, label=label2, linewidth=2)
        if show_std:
            axes[1].fill_between(x2, y2 - std2, y2 + std2, alpha=0.3, label='±1 Std Dev')
        axes[1].set_xlabel(x2_label)
        axes[1].set_ylabel(y2_label)
        axes[1].set_title(title2)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(path + "/" + filename)
        plt.close()

    def plot(self, path):
        self.plot_graph(x_values=np.array(self.timesteps),
                        y_values=np.array(self.rewards),
                        std_values=np.array([]),
                        x_label="Timesteps",
                        y_label="Rewards",
                        title=f"Running Episode",
                        label="Reward",
                        filename="episode_reward_plot.png",
                        show_std=False ,
                        path=path
                        )