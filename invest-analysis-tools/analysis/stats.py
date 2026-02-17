"""Statistical and technical analysis metrics module."""

from typing import Optional
import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of closing prices.
        period: RSI period (default 14).
        
    Returns:
        Series of RSI values.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_sma(prices: pd.Series, period: int = 200) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Series of closing prices.
        period: SMA period (default 200).
        
    Returns:
        Series of SMA values.
    """
    return prices.rolling(window=period).mean()


def calculate_volatility(returns: pd.Series, annualized: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        returns: Series of returns.
        annualized: If True, annualize the volatility (assuming 252 trading days).
        
    Returns:
        Volatility value.
    """
    volatility = returns.std()
    
    if annualized:
        volatility = volatility * np.sqrt(252)
    
    return volatility


def calculate_hurst_exponent(prices: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent to detect trend persistence.
    
    Hurst < 0.5: Mean-reverting (anti-persistent)
    Hurst = 0.5: Random walk
    Hurst > 0.5: Trending (persistent)
    
    Args:
        prices: Series of prices.
        max_lag: Maximum lag for calculation.
        
    Returns:
        Hurst exponent value.
    """
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
    
    # Calculate Hurst using log-log regression
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2.0
    
    return hurst


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio.
    
    Args:
        returns: Series of returns.
        risk_free_rate: Annual risk-free rate (default 2%).
        
    Returns:
        Sharpe Ratio (annualized).
    """
    if returns.std() == 0:
        return 0.0
    
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    
    if volatility == 0:
        return 0.0
    
    sharpe = excess_returns / volatility
    return sharpe


def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate Beta (sensitivity to market movements).
    
    Args:
        returns: Series of asset returns.
        market_returns: Series of market returns (e.g., S&P 500).
        
    Returns:
        Beta value.
    """
    # Align indices
    aligned = pd.DataFrame({'asset': returns, 'market': market_returns}).dropna()
    
    if len(aligned) < 2:
        return 1.0
    
    covariance = aligned['asset'].cov(aligned['market'])
    market_variance = aligned['market'].var()
    
    if market_variance == 0:
        return 1.0
    
    beta = covariance / market_variance
    return beta
