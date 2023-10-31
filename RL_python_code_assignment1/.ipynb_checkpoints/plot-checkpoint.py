# Inside plot.py (or the module where Plot is defined)
import matplotlib.pyplot as plt

class Plot:
    @staticmethod
    def plot_episode_length(labels, data):
        # Create a bar plot of episode length
        plt.figure(figsize=(8, 6))
        plt.bar(labels, data)
        plt.xlabel("Algorithm")
        plt.ylabel("Episode Length")
        plt.title("Episode Length Comparison")
        plt.show()
