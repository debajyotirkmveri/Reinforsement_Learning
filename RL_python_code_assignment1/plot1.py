import matplotlib.pyplot as plt

class Plot:
    @staticmethod
    def plot_rewards(labels, data):
        """
        Plot rewards data for different experiments.

        Args:
            labels (list): A list of labels for each data series.
            data (list): A list of reward data for each experiment.
        """
        plt.figure(figsize=(10, 6))
        for label, rewards in zip(labels, data):
            plt.plot(range(len(rewards)), rewards, label=label)
        plt.xlabel("Time Steps")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.title("Epsilon-Greedy Rewards")
        plt.grid(True)
        plt.show()
