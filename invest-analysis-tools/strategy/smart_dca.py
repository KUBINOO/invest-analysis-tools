"""
Quant Investment Framework: main entry point.
Orchestrates: S&P 500 load -> Fundamental screener -> Markowitz weights -> Smart DCA signals.
"""

import sys
from tabulate import tabulate

from data.loader import get_sp500_tickers, get_historical_data
from analysis.screener import FundamentalScreener
from optimization.markowitz import PortfolioOptimizer
from strategy.smart_dca import SmartDCAAnalyzer


def main() -> None:
    demo_limit = 200   # Limit first N tickers for demo speed
    top_stocks = 5     # Keep top N stocks from screener
    period = "2y"      # Historical data period

    print("=" * 60)
    print("  Quant Investment Framework")
    print("=" * 60)

    # 1. S&P 500 tickers (limited for demo)
    print("\n[1/5] Downloading S&P 500 ticker list...")
    try:
        all_tickers = get_sp500_tickers()
        
        # Safe slicing in case all_tickers is empty or shorter than limit
        limit = min(demo_limit, len(all_tickers))
        tickers_demo = all_tickers[:limit]
        
        print(f"      Loaded {len(all_tickers)} tickers. Using first {limit} for demo speed.")
        
    except Exception as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    # 2. Fundamental screener (Buffett style)
    print("\n[2/5] Running Fundamental Screener (Buffett style)...")
    
    # Initialize screener with strict criteria
    screener = FundamentalScreener(
        min_roe=0.15,            # Return on Equity > 15%
        max_debt_to_equity=1.0,  # Low debt
        min_profit_margin=0.10,  # Healthy margins
        max_peg=2.0,             # Reasonable valuation
        top_n=top_stocks,
    )
    
    # Run filter
    try:
        screened = screener.filter_stocks(tickers_demo)
    except Exception as e:
        print(f"      ERROR during screening: {e}")
        sys.exit(1)

    if screened.empty:
        print("      No stocks passed the filter criteria. Exiting.")
        sys.exit(1)

    selected_tickers = screened["Ticker"].tolist()
    
    # Print screening results table
    print(tabulate(screened, headers="keys", tablefmt="rounded_grid", showindex=False))
    print(f"\n      Selected TOP {len(selected_tickers)} stocks: {selected_tickers}")

    # 3. Historical prices for selected tickers
    print("\n[3/5] Downloading historical price data...")
    try:
        prices_df = get_historical_data(selected_tickers, period=period)
        
        if prices_df.empty or len(prices_df) < 30:
            print("      Insufficient price data downloaded. Exiting.")
            sys.exit(1)
            
        print(f"      Downloaded {len(prices_df)} days of data for {list(prices_df.columns)}")
        
    except Exception as e:
        print(f"      ERROR: {e}")
        sys.exit(1)

    # 4. Markowitz optimization
    print("\n[4/5] Portfolio Optimization (Markowitz – Max Sharpe)...")
    try:
        optimizer = PortfolioOptimizer(risk_free_rate=0.04) # 4% risk free rate (approx. US T-Bill)
        optimizer.set_prices(prices_df)
        weights = optimizer.optimize_sharpe_ratio()

        if not weights:
            print("      Optimization failed to converge or returned no weights.")
        else:
            # Sort weights for better readability
            weight_rows = [[t, f"{w*100:.2f} %"] for t, w in sorted(weights.items(), key=lambda x: -x[1])]
            print(tabulate(weight_rows, headers=["Ticker", "Allocated Weight"], tablefmt="rounded_grid"))
            print("\n      Recommendation: Portfolio allocation percentage based on Sharpe Ratio.")
            
    except Exception as e:
        print(f"      ERROR during optimization: {e}")

    # 5. Smart DCA for each selected ticker
    print("\n[5/5] Smart DCA Analysis – Current Action (Buy / Wait)...")
    dca = SmartDCAAnalyzer()
    
    for ticker in selected_tickers:
        if ticker not in prices_df.columns:
            continue
            
        # Get price series for specific ticker
        series = prices_df[ticker].dropna()
        if series.empty:
            continue
            
        current_price = float(series.iloc[-1])
        
        # Analyze
        try:
            action_key, message = dca.analyze_entry(ticker, current_price, series)
            print(f"      {message}")
        except Exception as e:
            print(f"      Could not analyze {ticker}: {e}")

    print("\n" + "=" * 60)
    print("  Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()