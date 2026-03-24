import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PlottingView:
    """
    Handles all plotting logic for the portfolio tracker.
    """

    @staticmethod
    def plot_prices(price_df: pd.DataFrame) -> None:
        """
        Plot raw price series.
        """
        if price_df.empty:
            print("No data to plot.")
            return

        plt.figure()
        price_df.plot(ax=plt.gca())
        plt.title("Historical Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_normalized_prices(price_df: pd.DataFrame) -> None:
        """
        Plot normalized price series (starting at 1).
        """
        if price_df.empty:
            print("No data to plot.")
            return

        normalized = price_df / price_df.iloc[0]

        plt.figure()
        normalized.plot(ax=plt.gca())
        plt.title("Normalized Prices (Start = 1)")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_simulation_paths(paths: np.ndarray, n_show: int = 100) -> None:
        """
        Plot Monte Carlo simulation paths.

        Args:
            paths: Array of shape (time, paths)
            n_show: Number of paths to display (to avoid clutter)
        """
        if paths.size == 0:
            print("No simulation data to plot.")
            return

        n_paths = paths.shape[1]
        n_show = min(n_show, n_paths)

        plt.figure()

        for i in range(n_show):
            plt.plot(paths[:, i], linewidth=0.8, alpha=0.6)

        plt.title(f"Monte Carlo Simulation ({n_show} paths shown)")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()