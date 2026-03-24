from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Asset:
    """
    Represents a single portfolio position.

    Attributes:
        ticker: Market ticker symbol, e.g. 'AAPL'.
        sector: Sector classification, e.g. 'Technology'.
        asset_class: Asset class, e.g. 'Equity'.
        quantity: Number of units held.
        purchase_price: Average purchase price per unit.
    """
    ticker: str
    sector: str
    asset_class: str
    quantity: float
    purchase_price: float

    def __post_init__(self) -> None:
        cleaned_ticker = self.ticker.strip().upper()
        cleaned_sector = self.sector.strip()
        cleaned_asset_class = self.asset_class.strip()

        if not cleaned_ticker:
            raise ValueError("Ticker cannot be empty.")
        if not cleaned_sector:
            raise ValueError("Sector cannot be empty.")
        if not cleaned_asset_class:
            raise ValueError("Asset class cannot be empty.")
        if self.quantity <= 0:
            raise ValueError("Quantity must be greater than 0.")
        if self.purchase_price <= 0:
            raise ValueError("Purchase price must be greater than 0.")

        object.__setattr__(self, "ticker", cleaned_ticker)
        object.__setattr__(self, "sector", cleaned_sector)
        object.__setattr__(self, "asset_class", cleaned_asset_class)

    @property
    def cost_basis(self) -> float:
        """Total amount invested in this asset."""
        return self.quantity * self.purchase_price

    def to_dict(self) -> dict:
        """Return a serializable dictionary representation of the asset."""
        return {
            "ticker": self.ticker,
            "sector": self.sector,
            "asset_class": self.asset_class,
            "quantity": self.quantity,
            "purchase_price": self.purchase_price,
            "cost_basis": self.cost_basis,
        }