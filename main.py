import math
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# =========================
# MODEL
# =========================
@dataclass
class Asset:
    ticker: str
    sector: str
    asset_class: str
    quantity: float
    purchase_price: float

    @property
    def transaction_value(self) -> float:
        return self.quantity * self.purchase_price


class Portfolio:
    def __init__(self) -> None:
        self.assets: List[Asset] = []

    def add_asset(
        self,
        ticker: str,
        sector: str,
        asset_class: str,
        quantity: float,
        purchase_price: float
    ) -> None:
        self.assets.append(
            Asset(
                ticker=ticker.upper(),
                sector=sector,
                asset_class=asset_class,
                quantity=quantity,
                purchase_price=purchase_price
            )
        )

    def get_assets(self) -> List[Asset]:
        return self.assets

    def tickers(self) -> List[str]:
        return sorted(list({asset.ticker for asset in self.assets}))

    def compute_portfolio_table(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        rows = []

        for asset in self.assets:
            current_price = current_prices.get(asset.ticker, np.nan)
            current_value = asset.quantity * current_price if pd.notna(current_price) else np.nan

            rows.append({
                "Ticker": asset.ticker,
                "Sector": asset.sector,
                "Asset Class": asset.asset_class,
                "Quantity": asset.quantity,
                "Purchase Price": asset.purchase_price,
                "Transaction Value": asset.transaction_value,
                "Current Price": current_price,
                "Current Value": current_value,
            })

        return pd.DataFrame(rows)

    def total_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        total = 0.0
        for asset in self.assets:
            price = current_prices.get(asset.ticker)
            if price is not None and pd.notna(price):
                total += asset.quantity * price
        return total

    def compute_weights(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        df = self.compute_portfolio_table(current_prices)
        total_value = df["Current Value"].sum()

        if total_value > 0:
            df["Weight"] = df["Current Value"] / total_value
        else:
            df["Weight"] = np.nan

        return df[["Ticker", "Sector", "Asset Class", "Current Value", "Weight"]]

    def compute_group_weights(self, current_prices: Dict[str, float], group_col: str) -> pd.DataFrame:
        df = self.compute_portfolio_table(current_prices)
        grouped = df.groupby(group_col, as_index=False)["Current Value"].sum()
        total_value = grouped["Current Value"].sum()

        if total_value > 0:
            grouped["Weight"] = grouped["Current Value"] / total_value
        else:
            grouped["Weight"] = np.nan

        return grouped

    def compute_return_contributions(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        df = self.compute_portfolio_table(current_prices).copy()
        df["PnL"] = df["Current Value"] - df["Transaction Value"]
        df["Return %"] = np.where(
            df["Transaction Value"] > 0,
            (df["PnL"] / df["Transaction Value"]) * 100,
            np.nan
        )
        return df[["Ticker", "Transaction Value", "Current Value", "PnL", "Return %"]]


# =========================
# VIEW
# =========================
class PortfolioView:
    @staticmethod
    def show_message(message: str) -> None:
        print(f"\n{message}")

    @staticmethod
    def show_dataframe(df: pd.DataFrame, title: Optional[str] = None) -> None:
        if title:
            print(f"\n{title}")
            print("-" * len(title))

        if df.empty:
            print("No data available.")
            return

        display_df = df.copy()

        for col in display_df.columns:
            if pd.api.types.is_float_dtype(display_df[col]):
                if "Weight" in col:
                    display_df[col] = display_df[col].map(
                        lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
                    )
                elif "%" in col:
                    display_df[col] = display_df[col].map(
                        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                    )
                else:
                    display_df[col] = display_df[col].map(
                        lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A"
                    )

        print(display_df.to_string(index=False))

    @staticmethod
    def plot_prices(price_df: pd.DataFrame, tickers: List[str]) -> None:
        if price_df.empty:
            print("No historical data to plot.")
            return

        plt.figure(figsize=(11, 6))
        for ticker in tickers:
            if ticker in price_df.columns:
                plt.plot(price_df.index, price_df[ticker], label=ticker)

        plt.title("Historical Adjusted Close Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_normalized_prices(price_df: pd.DataFrame, tickers: List[str]) -> None:
        if price_df.empty:
            print("No historical data to plot.")
            return

        normalized = price_df.copy()
        for col in normalized.columns:
            first_valid = normalized[col].dropna()
            if not first_valid.empty:
                normalized[col] = normalized[col] / first_valid.iloc[0] * 100

        plt.figure(figsize=(11, 6))
        for ticker in tickers:
            if ticker in normalized.columns:
                plt.plot(normalized.index, normalized[ticker], label=ticker)

        plt.title("Normalized Price Performance (Start = 100)")
        plt.xlabel("Date")
        plt.ylabel("Indexed Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_simulation(paths_subset: np.ndarray) -> None:
        plt.figure(figsize=(11, 6))
        plt.plot(paths_subset, alpha=0.35)
        plt.title("Monte Carlo Simulation of Portfolio Value")
        plt.xlabel("Month")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# =========================
# CONTROLLER
# =========================
class PortfolioController:
    def __init__(self, portfolio: Portfolio, view: PortfolioView) -> None:
        self.portfolio = portfolio
        self.view = view

    def add_asset_interactive(self) -> None:
        try:
            ticker = input("Ticker: ").strip().upper()
            sector = input("Sector: ").strip()
            asset_class = input("Asset class: ").strip()
            quantity = float(input("Quantity: ").strip())
            purchase_price = float(input("Purchase price: ").strip())

            if quantity <= 0 or purchase_price <= 0:
                self.view.show_message("Quantity and purchase price must be positive.")
                return

            self.portfolio.add_asset(ticker, sector, asset_class, quantity, purchase_price)
            self.view.show_message(f"Asset {ticker} added successfully.")

        except ValueError:
            self.view.show_message("Invalid input. Please enter numeric values for quantity and purchase price.")

    def fetch_current_prices(self) -> Dict[str, float]:
        tickers = self.portfolio.tickers()
        if not tickers:
            return {}

        prices: Dict[str, float] = {}

        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
                if not data.empty:
                    prices[ticker] = float(data["Close"].dropna().iloc[-1])
                else:
                    prices[ticker] = np.nan
            except Exception:
                prices[ticker] = np.nan

        return prices

    def fetch_historical_prices(self, period: str = "5y") -> pd.DataFrame:
        tickers = self.portfolio.tickers()
        if not tickers:
            return pd.DataFrame()

        try:
            data = yf.download(
                tickers=tickers,
                period=period,
                auto_adjust=True,
                progress=False
            )

            if data.empty:
                return pd.DataFrame()

            if isinstance(data.columns, pd.MultiIndex):
                close_df = data["Close"].copy()
            else:
                close_df = data[["Close"]].copy()
                close_df.columns = tickers

            close_df = close_df.dropna(how="all")
            return close_df

        except Exception:
            return pd.DataFrame()

    def show_portfolio(self) -> None:
        current_prices = self.fetch_current_prices()
        df = self.portfolio.compute_portfolio_table(current_prices)
        self.view.show_dataframe(df, "Current Portfolio")

        total_value = self.portfolio.total_portfolio_value(current_prices)
        self.view.show_message(f"Total portfolio value: {total_value:,.2f}")

    def show_weights(self) -> None:
        current_prices = self.fetch_current_prices()

        asset_weights = self.portfolio.compute_weights(current_prices)
        sector_weights = self.portfolio.compute_group_weights(current_prices, "Sector")
        class_weights = self.portfolio.compute_group_weights(current_prices, "Asset Class")

        self.view.show_dataframe(asset_weights, "Asset Weights")
        self.view.show_dataframe(sector_weights, "Sector Weights")
        self.view.show_dataframe(class_weights, "Asset Class Weights")

    def show_profit_and_loss(self) -> None:
        current_prices = self.fetch_current_prices()
        df = self.portfolio.compute_return_contributions(current_prices)
        self.view.show_dataframe(df, "Profit / Loss Overview")

    def show_prices(self) -> None:
        if not self.portfolio.tickers():
            self.view.show_message("No assets in portfolio.")
            return

        price_df = self.fetch_historical_prices(period="5y")

        if price_df.empty:
            self.view.show_message("Could not retrieve historical prices.")
            return

        self.view.show_dataframe(price_df.tail(), "Historical Price Snapshot")

        user_input = input(
            "Enter tickers to plot separated by commas, or press Enter for all: "
        ).strip().upper()

        if user_input:
            selected = [x.strip() for x in user_input.split(",") if x.strip()]
            selected = [x for x in selected if x in price_df.columns]
        else:
            selected = list(price_df.columns)

        if not selected:
            self.view.show_message("No valid tickers selected.")
            return

        self.view.plot_prices(price_df[selected], selected)
        self.view.plot_normalized_prices(price_df[selected], selected)

    def run_simulation(self, years: int = 15, n_paths: int = 100_000) -> None:
        tickers = self.portfolio.tickers()
        if not tickers:
            self.view.show_message("Portfolio is empty.")
            return

        current_prices = self.fetch_current_prices()
        current_table = self.portfolio.compute_portfolio_table(current_prices)
        initial_value = current_table["Current Value"].sum()

        if initial_value <= 0 or pd.isna(initial_value):
            self.view.show_message("Could not determine current portfolio value.")
            return

        price_df = self.fetch_historical_prices(period="5y")
        if price_df.empty:
            self.view.show_message("Could not retrieve historical prices for simulation.")
            return

        price_df = price_df.dropna(axis=1, how="all")
        price_df = price_df.dropna(how="any")

        usable_tickers = [t for t in tickers if t in price_df.columns]
        if not usable_tickers:
            self.view.show_message("No usable tickers found for simulation.")
            return

        returns = np.log(price_df[usable_tickers] / price_df[usable_tickers].shift(1)).dropna()
        if returns.empty:
            self.view.show_message("Not enough return history for simulation.")
            return

        current_table = current_table[current_table["Ticker"].isin(usable_tickers)].copy()
        current_table = current_table.dropna(subset=["Current Value"])
        total_value = current_table["Current Value"].sum()

        if total_value <= 0:
            self.view.show_message("Could not compute portfolio weights for simulation.")
            return

        weights = (
            current_table.set_index("Ticker")["Current Value"] / total_value
        ).reindex(usable_tickers).fillna(0.0).values

        mean_daily = returns.mean().values
        cov_daily = returns.cov().values

        trading_days = 252
        mu_annual = mean_daily * trading_days
        cov_annual = cov_daily * trading_days

        portfolio_mu = float(weights @ mu_annual)
        portfolio_sigma = float(np.sqrt(weights @ cov_annual @ weights))

        steps = years * 12
        dt = 1 / 12

        rng = np.random.default_rng(42)
        shocks = rng.normal(0, 1, size=(steps, n_paths))

        paths = np.zeros((steps + 1, n_paths))
        paths[0, :] = initial_value

        drift = (portfolio_mu - 0.5 * portfolio_sigma**2) * dt
        diffusion_scale = portfolio_sigma * math.sqrt(dt)

        for t in range(1, steps + 1):
            paths[t, :] = paths[t - 1, :] * np.exp(drift + diffusion_scale * shocks[t - 1, :])

        terminal_values = paths[-1, :]
        summary = pd.DataFrame({
            "Statistic": [
                "Initial Value",
                "Expected Annual Return (mu)",
                "Annual Volatility (sigma)",
                "Mean Terminal Value",
                "Median Terminal Value",
                "5th Percentile",
                "95th Percentile"
            ],
            "Value": [
                initial_value,
                portfolio_mu,
                portfolio_sigma,
                np.mean(terminal_values),
                np.median(terminal_values),
                np.percentile(terminal_values, 5),
                np.percentile(terminal_values, 95)
            ]
        })

        self.view.show_dataframe(summary, "Monte Carlo Simulation Summary")
        self.view.plot_simulation(paths[:, :100])

    def menu(self) -> None:
        while True:
            print("\nPortfolio Tracker")
            print("1. Add asset")
            print("2. Show portfolio")
            print("3. Show weights")
            print("4. Show prices and charts")
            print("5. Run 15-year simulation")
            print("6. Show profit/loss")
            print("7. Load demo portfolio")
            print("8. Exit")

            choice = input("Choose an option: ").strip()

            if choice == "1":
                self.add_asset_interactive()
            elif choice == "2":
                self.show_portfolio()
            elif choice == "3":
                self.show_weights()
            elif choice == "4":
                self.show_prices()
            elif choice == "5":
                self.run_simulation()
            elif choice == "6":
                self.show_profit_and_loss()
            elif choice == "7":
                self.load_demo_portfolio()
            elif choice == "8":
                self.view.show_message("Exiting application.")
                break
            else:
                self.view.show_message("Invalid option. Please try again.")

    def load_demo_portfolio(self) -> None:
        if self.portfolio.get_assets():
            self.view.show_message("Portfolio already contains assets.")
            return

        demo_assets = [
            ("AAPL", "Technology", "Equity", 10, 180.0),
            ("MSFT", "Technology", "Equity", 8, 320.0),
            ("JNJ", "Healthcare", "Equity", 12, 150.0),
            ("XOM", "Energy", "Equity", 15, 105.0),
        ]

        for ticker, sector, asset_class, quantity, purchase_price in demo_assets:
            self.portfolio.add_asset(ticker, sector, asset_class, quantity, purchase_price)

        self.view.show_message("Demo portfolio loaded.")


# =========================
# MAIN
# =========================
def main() -> None:
    portfolio = Portfolio()
    view = PortfolioView()
    controller = PortfolioController(portfolio, view)
    controller.menu()


if __name__ == "__main__":
    main()
