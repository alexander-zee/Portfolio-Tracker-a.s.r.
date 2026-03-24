import pandas as pd
import yfinance as yf


class MarketDataService:
    """
    Service class for retrieving market data from Yahoo Finance.
    """

    @staticmethod
    def get_current_prices(tickers: list[str]) -> dict[str, float]:
        """
        Retrieve the most recent closing price for each ticker.

        Args:
            tickers: List of ticker symbols.

        Returns:
            Dictionary mapping ticker symbols to current prices.
            If a price cannot be retrieved, the value is NaN.
        """
        if not tickers:
            return {}

        unique_tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        if not unique_tickers:
            return {}

        try:
            data = yf.download(
                tickers=unique_tickers,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            return {ticker: float("nan") for ticker in unique_tickers}

        prices: dict[str, float] = {}

        if len(unique_tickers) == 1:
            ticker = unique_tickers[0]
            try:
                close_series = data["Close"].dropna()
                prices[ticker] = float(close_series.iloc[-1]) if not close_series.empty else float("nan")
            except Exception:
                prices[ticker] = float("nan")
            return prices

        for ticker in unique_tickers:
            try:
                close_series = data[ticker]["Close"].dropna()
                prices[ticker] = float(close_series.iloc[-1]) if not close_series.empty else float("nan")
            except Exception:
                prices[ticker] = float("nan")

        return prices

    @staticmethod
    def get_historical_prices(
        tickers: list[str],
        period: str = "5y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Retrieve historical adjusted closing prices for a list of tickers.

        Args:
            tickers: List of ticker symbols.
            period: Lookback period accepted by yfinance, e.g. '1y', '5y'.
            interval: Data interval, e.g. '1d', '1wk', '1mo'.

        Returns:
            DataFrame with dates as index and tickers as columns.
            Missing or invalid tickers are kept as columns with NaN values where needed.
        """
        if not tickers:
            return pd.DataFrame()

        unique_tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        if not unique_tickers:
            return pd.DataFrame()

        try:
            data = yf.download(
                tickers=unique_tickers,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            return pd.DataFrame(columns=unique_tickers)

        if data.empty:
            return pd.DataFrame(columns=unique_tickers)

        try:
            if len(unique_tickers) == 1:
                ticker = unique_tickers[0]
                if "Close" not in data:
                    return pd.DataFrame(columns=unique_tickers)

                result = data[["Close"]].copy()
                result.columns = [ticker]
                return result.dropna(how="all")

            frames = []
            for ticker in unique_tickers:
                try:
                    close_series = data[ticker]["Close"].rename(ticker)
                    frames.append(close_series)
                except Exception:
                    frames.append(pd.Series(name=ticker, dtype=float))

            result = pd.concat(frames, axis=1)
            return result.dropna(how="all")

        except Exception:
            return pd.DataFrame(columns=unique_tickers)