import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for portfolio value projection using GBM.
    """

    @staticmethod
    def estimate_parameters(
        price_df: pd.DataFrame,
        weights: np.ndarray,
    ) -> tuple[float, float]:
        """
        Estimate portfolio drift (mu) and volatility (sigma).

        Args:
            price_df: DataFrame of historical prices (columns = assets).
            weights: Portfolio weights (sum to 1).

        Returns:
            (mu, sigma) annualized.
        """
        log_returns = np.log(price_df / price_df.shift(1)).dropna()

        mu_vector = log_returns.mean() * 252
        cov_matrix = log_returns.cov() * 252

        mu_portfolio = float(weights @ mu_vector)
        sigma_portfolio = float(np.sqrt(weights @ cov_matrix @ weights))

        return mu_portfolio, sigma_portfolio

    @staticmethod
    def simulate_gbm(
        initial_value: float,
        mu: float,
        sigma: float,
        years: int = 15,
        n_paths: int = 100_000,
        steps_per_year: int = 12,
        seed: int | None = 42,
    ) -> np.ndarray:
        """
        Simulate portfolio value paths using Geometric Brownian Motion.

        Returns:
            Array of shape (n_steps + 1, n_paths)
        """
        if initial_value <= 0:
            raise ValueError("Initial value must be positive.")

        if sigma < 0:
            raise ValueError("Volatility must be non-negative.")

        if seed is not None:
            np.random.seed(seed)

        n_steps = years * steps_per_year
        dt = 1 / steps_per_year

        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = initial_value

        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_paths)

            paths[t] = paths[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt
                + sigma * np.sqrt(dt) * z
            )

        return paths

    @staticmethod
    def summarize(paths: np.ndarray) -> dict:
        """
        Compute summary statistics of simulated paths.

        Returns:
            Dictionary with key metrics.
        """
        final_values = paths[-1]

        return {
            "expected_final": float(np.mean(final_values)),
            "median_final": float(np.median(final_values)),
            "p5": float(np.percentile(final_values, 5)),
            "p95": float(np.percentile(final_values, 95)),
            "prob_loss": float(np.mean(final_values < paths[0, 0])),
        }

    @staticmethod
    def run_simulation(
        price_df: pd.DataFrame,
        weights: np.ndarray,
        initial_value: float,
        years: int = 15,
        n_paths: int = 100_000,
    ) -> tuple[np.ndarray, dict]:
        """
        Full pipeline: estimate parameters, simulate, summarize.
        """
        if price_df.empty:
            raise ValueError("Price data is empty.")

        if len(weights) != price_df.shape[1]:
            raise ValueError("Weights must match number of assets.")

        if not np.isclose(np.sum(weights), 1):
            raise ValueError("Weights must sum to 1.")

        price_df = price_df.dropna(how="any")

        if price_df.shape[0] < 2:
            raise ValueError("Not enough data to compute returns.")

        mu, sigma = MonteCarloSimulator.estimate_parameters(price_df, weights)

        paths = MonteCarloSimulator.simulate_gbm(
            initial_value=initial_value,
            mu=mu,
            sigma=sigma,
            years=years,
            n_paths=n_paths,
        )

        summary = MonteCarloSimulator.summarize(paths)

        return paths, summary