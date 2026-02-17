"""Mean-Variance (Markowitz) portfolio optimization."""

from typing import Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Mean-Variance optimizer: maximizes Sharpe ratio (efficient frontier).
    Long-only, fully invested portfolio.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: Annual risk-free rate (e.g. 0.02 = 2%).
        """
        self.risk_free_rate = risk_free_rate
        self._prices: pd.DataFrame = pd.DataFrame()
        self._returns: pd.DataFrame = pd.DataFrame()
        self._cov_matrix: np.ndarray = np.array([])
        self._mean_returns: np.ndarray = np.array([])

    def set_prices(self, prices_df: pd.DataFrame) -> None:
        """
        Set price DataFrame (columns = tickers, index = dates).

        Args:
            prices_df: Adjusted close (or close) prices.
        """
        self._prices = prices_df.dropna(how="all").copy()
        self._returns = self._prices.pct_change().dropna()
        self._mean_returns = self._returns.mean().values * 252
        self._cov_matrix = self._returns.cov().values * 252

    def _negative_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe (for minimization)."""
        port_return = np.dot(weights, self._mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self._cov_matrix, weights)))
        if port_vol <= 0:
            return 0.0
        sharpe = (port_return - self.risk_free_rate) / port_vol
        return -sharpe

    def optimize_sharpe_ratio(self) -> Dict[str, float]:
        """
        Find long-only weights that maximize Sharpe ratio.

        Returns:
            Dict mapping ticker -> optimal weight (sum = 1.0, each in [0, 1]).
        """
        if self._returns.empty or self._cov_matrix.size == 0:
            return {}

        n = len(self._returns.columns)
        tickers = list(self._returns.columns)

        # Bounds: each weight in [0, 1]
        bounds = tuple((0.0, 1.0) for _ in range(n))
        # Constraint: sum of weights = 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        # Initial guess: equal weight
        x0 = np.ones(n) / n

        result = minimize(
            self._negative_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            # Fallback to equal weight if optimization fails
            weights = np.ones(n) / n
        else:
            weights = result.x
            weights = np.maximum(weights, 0.0)
            weights /= weights.sum()

        return dict(zip(tickers, weights.tolist()))
