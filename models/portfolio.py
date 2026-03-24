from __future__ import annotations

from typing import Iterable
import pandas as pd

from models.asset import Asset


class Portfolio:
    """
    Represents a portfolio of assets and provides portfolio-level analytics.
    """

    def __init__(self, assets: Iterable[Asset] | None = None) -> None:
        self._assets: list[Asset] = list(assets) if assets is not None else []

    def add_asset(self, asset: Asset) -> None:
        self._assets.append(asset)

    def remove_asset(self, ticker: str) -> None:
        ticker = ticker.strip().upper()
        original_count = len(self._assets)
        self._assets = [a for a in self._assets if a.ticker != ticker]

        if len(self._assets) == original_count:
            raise ValueError(f"No asset with ticker '{ticker}' found.")

    def get_assets(self) -> list[Asset]:
        return list(self._assets)

    def is_empty(self) -> bool:
        return len(self._assets) == 0

    def total_cost_basis(self) -> float:
        return sum(a.cost_basis for a in self._assets)

    def to_dataframe(self) -> pd.DataFrame:
        if self.is_empty():
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "sector",
                    "asset_class",
                    "quantity",
                    "purchase_price",
                    "cost_basis",
                ]
            )
        return pd.DataFrame(a.to_dict() for a in self._assets)

    def build_portfolio_table(self, current_prices: dict[str, float]) -> pd.DataFrame:
        if self.is_empty():
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "sector",
                    "asset_class",
                    "quantity",
                    "purchase_price",
                    "cost_basis",
                    "current_price",
                    "current_value",
                    "pnl_eur",
                    "pnl_pct",
                    "weight",
                ]
            )

        rows = []
        for a in self._assets:
            price = current_prices.get(a.ticker)

            if price is None or pd.isna(price):
                current_value = float("nan")
                pnl_eur = float("nan")
                pnl_pct = float("nan")
            else:
                current_value = a.quantity * price
                pnl_eur = current_value - a.cost_basis
                pnl_pct = pnl_eur / a.cost_basis

            rows.append(
                {
                    "ticker": a.ticker,
                    "sector": a.sector,
                    "asset_class": a.asset_class,
                    "quantity": a.quantity,
                    "purchase_price": a.purchase_price,
                    "cost_basis": a.cost_basis,
                    "current_price": price,
                    "current_value": current_value,
                    "pnl_eur": pnl_eur,
                    "pnl_pct": pnl_pct,
                }
            )

        df = pd.DataFrame(rows)

        total_value = df["current_value"].sum(skipna=True)
        if total_value > 0:
            df["weight"] = df["current_value"] / total_value
        else:
            df["weight"] = float("nan")

        return df

    def aggregate_by_sector(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        return self._aggregate(portfolio_df, "sector")

    def aggregate_by_asset_class(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        return self._aggregate(portfolio_df, "asset_class")

    def _aggregate(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[column, "current_value", "weight"])

        grouped = (
            df.groupby(column, dropna=False, as_index=False)["current_value"]
            .sum()
            .sort_values("current_value", ascending=False)
        )

        total = grouped["current_value"].sum(skipna=True)
        if total > 0:
            grouped["weight"] = grouped["current_value"] / total
        else:
            grouped["weight"] = float("nan")

        return grouped