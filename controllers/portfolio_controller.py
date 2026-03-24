import pandas as pd

from models.asset import Asset
from models.portfolio import Portfolio
from services.market_data import MarketDataService
from services.simulation import MonteCarloSimulator
from views.cli import CLIView
from views.plotting import PlottingView


class PortfolioController:
    """
    Controller that connects the model, services, and views.
    """

    def __init__(self) -> None:
        self.portfolio = Portfolio()
        self.cli_view = CLIView()
        self.plotting_view = PlottingView()

    def add_asset(self) -> None:
        try:
            ticker = self.cli_view.prompt_string("Enter ticker: ").upper()
            sector = self.cli_view.prompt_string("Enter sector: ")
            asset_class = self.cli_view.prompt_string("Enter asset class: ")
            quantity = self.cli_view.prompt_float("Enter quantity: ")
            purchase_price = self.cli_view.prompt_float("Enter purchase price: ")

            asset = Asset(
                ticker=ticker,
                sector=sector,
                asset_class=asset_class,
                quantity=quantity,
                purchase_price=purchase_price,
            )

            self.portfolio.add_asset(asset)
            self.cli_view.show_message(f"Added {ticker} to portfolio.")

        except ValueError as e:
            self.cli_view.show_message(f"Input error: {e}")

    def remove_asset(self) -> None:
        try:
            ticker = self.cli_view.prompt_string("Enter ticker to remove: ").upper()
            self.portfolio.remove_asset(ticker)
            self.cli_view.show_message(f"Removed {ticker} from portfolio.")
        except ValueError as e:
            self.cli_view.show_message(str(e))

    def show_portfolio(self) -> None:
        if self.portfolio.is_empty():
            self.cli_view.show_message("Portfolio is empty.")
            return

        tickers = [asset.ticker for asset in self.portfolio.get_assets()]
        current_prices = MarketDataService.get_current_prices(tickers)
        portfolio_df = self.portfolio.build_portfolio_table(current_prices)

        self.cli_view.show_dataframe(portfolio_df, title="Portfolio Overview")

    def show_sector_allocation(self) -> None:
        if self.portfolio.is_empty():
            self.cli_view.show_message("Portfolio is empty.")
            return

        tickers = [asset.ticker for asset in self.portfolio.get_assets()]
        current_prices = MarketDataService.get_current_prices(tickers)
        portfolio_df = self.portfolio.build_portfolio_table(current_prices)
        sector_df = self.portfolio.aggregate_by_sector(portfolio_df)

        self.cli_view.show_dataframe(sector_df, title="Sector Allocation")

    def show_asset_class_allocation(self) -> None:
        if self.portfolio.is_empty():
            self.cli_view.show_message("Portfolio is empty.")
            return

        tickers = [asset.ticker for asset in self.portfolio.get_assets()]
        current_prices = MarketDataService.get_current_prices(tickers)
        portfolio_df = self.portfolio.build_portfolio_table(current_prices)
        asset_class_df = self.portfolio.aggregate_by_asset_class(portfolio_df)

        self.cli_view.show_dataframe(asset_class_df, title="Asset Class Allocation")

    def plot_historical_prices(self) -> None:
        if self.portfolio.is_empty():
            self.cli_view.show_message("Portfolio is empty.")
            return

        tickers = [asset.ticker for asset in self.portfolio.get_assets()]
        price_df = MarketDataService.get_historical_prices(tickers, period="5y", interval="1d")

        if price_df.empty:
            self.cli_view.show_message("No historical price data available.")
            return

        self.plotting_view.plot_prices(price_df)

    def plot_normalized_prices(self) -> None:
        if self.portfolio.is_empty():
            self.cli_view.show_message("Portfolio is empty.")
            return

        tickers = [asset.ticker for asset in self.portfolio.get_assets()]
        price_df = MarketDataService.get_historical_prices(tickers, period="5y", interval="1d")

        if price_df.empty:
            self.cli_view.show_message("No historical price data available.")
            return

        self.plotting_view.plot_normalized_prices(price_df)

    def run_monte_carlo_simulation(self) -> None:
        if self.portfolio.is_empty():
            self.cli_view.show_message("Portfolio is empty.")
            return

        tickers = [asset.ticker for asset in self.portfolio.get_assets()]
        current_prices = MarketDataService.get_current_prices(tickers)
        portfolio_df = self.portfolio.build_portfolio_table(current_prices)

        valid_df = portfolio_df.dropna(subset=["current_value"])
        if valid_df.empty:
            self.cli_view.show_message("No valid current prices available for simulation.")
            return

        initial_value = float(valid_df["current_value"].sum())
        weights = (valid_df["current_value"] / initial_value).to_numpy()

        valid_tickers = valid_df["ticker"].tolist()
        price_df = MarketDataService.get_historical_prices(valid_tickers, period="5y", interval="1d")

        if price_df.empty:
            self.cli_view.show_message("No historical price data available for simulation.")
            return

        try:
            paths, summary = MonteCarloSimulator.run_simulation(
                price_df=price_df,
                weights=weights,
                initial_value=initial_value,
                years=15,
                n_paths=100_000,
            )

            summary_df = pd.DataFrame(
                {
                    "Metric": list(summary.keys()),
                    "Value": list(summary.values()),
                }
            )

            self.cli_view.show_dataframe(summary_df, title="Monte Carlo Summary")
            self.plotting_view.plot_simulation_paths(paths)

        except ValueError as e:
            self.cli_view.show_message(f"Simulation error: {e}")

    def run(self) -> None:
        while True:
            self.cli_view.show_menu()
            choice = self.cli_view.prompt_string("Choose an option: ")

            if choice == "1":
                self.add_asset()
            elif choice == "2":
                self.remove_asset()
            elif choice == "3":
                self.show_portfolio()
            elif choice == "4":
                self.show_sector_allocation()
            elif choice == "5":
                self.show_asset_class_allocation()
            elif choice == "6":
                self.plot_historical_prices()
            elif choice == "7":
                self.plot_normalized_prices()
            elif choice == "8":
                self.run_monte_carlo_simulation()
            elif choice == "9":
                self.cli_view.show_message("Exiting portfolio tracker.")
                break
            else:
                self.cli_view.show_message("Invalid choice. Please try again.")