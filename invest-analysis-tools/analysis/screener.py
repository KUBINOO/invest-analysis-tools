"""Fundamental stock screener (Buffett-style criteria)."""

from typing import List, Optional
import pandas as pd
import yfinance as yf


class FundamentalScreener:
    """
    Screens stocks based on fundamental metrics (ROE, Debt/Equity, margins, PEG).
    """

    def __init__(
        self,
        min_roe: float = 0.15,
        max_debt_to_equity: float = 1.0,
        min_profit_margin: float = 0.10,
        max_peg: float = 2.0,
        top_n: int = 15,
    ):
        """
        Initialize screener with criteria thresholds.

        Args:
            min_roe: Minimum return on equity (e.g. 0.15 = 15%).
            max_debt_to_equity: Maximum debt-to-equity ratio.
            min_profit_margin: Minimum profit margin (e.g. 0.10 = 10%).
            max_peg: Maximum trailing PEG ratio (valuation).
            top_n: Maximum number of stocks to return (for performance).
        """
        self.min_roe = min_roe
        self.max_debt_to_equity = max_debt_to_equity
        self.min_profit_margin = min_profit_margin
        self.max_peg = max_peg
        self.top_n = top_n

    def _safe_float(self, value, default: Optional[float] = None) -> Optional[float]:
        """Convert value to float, return default if invalid."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_metrics(self, ticker: str) -> Optional[dict]:
        """Fetch info from yfinance and extract metrics for one ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            roe = self._safe_float(info.get("returnOnEquity"))
            debt_eq = self._safe_float(info.get("debtToEquity"))
            profit_margin = self._safe_float(info.get("profitMargins"))
            peg = self._safe_float(info.get("trailingPegRatio"))
            # Fallback for profit margin
            if profit_margin is None:
                profit_margin = self._safe_float(info.get("operatingMargins"))

            return {
                "Ticker": ticker,
                "ROE": roe,
                "DebtToEquity": debt_eq,
                "ProfitMargin": profit_margin,
                "PEG": peg,
            }
        except Exception:
            return None

    def _score_row(self, row: pd.Series) -> float:
        """Compute a simple quality score (higher = better)."""
        score = 0.0
        if row.get("ROE") is not None and row["ROE"] >= self.min_roe:
            score += 1.0 + min(row["ROE"] - self.min_roe, 0.2)
        if row.get("DebtToEquity") is not None and row["DebtToEquity"] <= self.max_debt_to_equity:
            score += 1.0
        if row.get("ProfitMargin") is not None and row["ProfitMargin"] >= self.min_profit_margin:
            score += 1.0
        if row.get("PEG") is not None and row["PEG"] < self.max_peg and row["PEG"] > 0:
            score += 1.0
        return score

    def filter_stocks(self, tickers: List[str]) -> pd.DataFrame:
        """
        Filter tickers by fundamental criteria and return DataFrame with metrics and Score.

        Args:
            tickers: List of ticker symbols.

        Returns:
            DataFrame with columns: Ticker, ROE, DebtToEquity, ProfitMargin, PEG, Score.
            Sorted by Score descending, limited to top_n rows.
        """
        rows = []
        for ticker in tickers:
            m = self._get_metrics(ticker)
            if m is not None:
                rows.append(m)

        if not rows:
            return pd.DataFrame(columns=["Ticker", "ROE", "DebtToEquity", "ProfitMargin", "PEG", "Score"])

        df = pd.DataFrame(rows)

        # Apply filters: keep only rows that pass all numeric criteria (where available)
        mask = pd.Series(True, index=df.index)
        if "ROE" in df.columns:
            mask &= (df["ROE"].isna()) | (df["ROE"] >= self.min_roe)
        if "DebtToEquity" in df.columns:
            mask &= (df["DebtToEquity"].isna()) | (df["DebtToEquity"] <= self.max_debt_to_equity)
        if "ProfitMargin" in df.columns:
            mask &= (df["ProfitMargin"].isna()) | (df["ProfitMargin"] >= self.min_profit_margin)
        if "PEG" in df.columns:
            mask &= (df["PEG"].isna()) | ((df["PEG"] > 0) & (df["PEG"] < self.max_peg))

        df = df.loc[mask].copy()
        df["Score"] = df.apply(self._score_row, axis=1)
        df = df.sort_values("Score", ascending=False).head(self.top_n).reset_index(drop=True)
        return df
