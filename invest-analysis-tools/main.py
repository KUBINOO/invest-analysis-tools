"""
Quant Investment Framework – main entry point.
Orchestrates: S&P 500 load -> Fundamental screener -> Markowitz weights -> Smart DCA signals.
"""

import sys
from tabulate import tabulate

from data.loader import get_sp500_tickers, get_historical_data
from analysis.screener import FundamentalScreener
from optimization.markowitz import PortfolioOptimizer
from strategy.smart_dca import SmartDCAAnalyzer


def main() -> None:
    demo_limit = 200  # First N S&P 500 tickers for demo speed
    top_stocks = 5   # Top N from screener
    period = "2y"

    print("=" * 60)
    print("  Quant Investment Framework")
    print("=" * 60)

    # 1. S&P 500 tickers (limited for demo)
    print("\n[1/5] Stahování seznamu S&P 500 tickerů...")
    try:
        all_tickers = get_sp500_tickers()
        tickers_demo = all_tickers[:demo_limit]
        print(f"      Načteno {len(all_tickers)} tickerů, pro demo použito prvních {demo_limit}.")
    except Exception as e:
        print(f"      CHYBA: {e}")
        sys.exit(1)

    # 2. Fundamental screener (Buffett style)
    print("\n[2/5] Fundamentální filtr (Buffett style)...")
    screener = FundamentalScreener(
        min_roe=0.15,
        max_debt_to_equity=1.0,
        min_profit_margin=0.10,
        max_peg=2.0,
        top_n=top_stocks,
    )
    screened = screener.filter_stocks(tickers_demo)

    if screened.empty:
        print("      Žádné akcie neprošly filtrem. Ukončuji.")
        sys.exit(1)

    selected_tickers = screened["Ticker"].tolist()
    print(tabulate(screened, headers="keys", tablefmt="rounded_grid", showindex=False))
    print(f"\n      Vybráno TOP {len(selected_tickers)} akcií: {selected_tickers}")

    # 3. Historical prices for selected tickers
    print("\n[3/5] Stahování historie cen...")
    try:
        prices_df = get_historical_data(selected_tickers, period=period)
        if prices_df.empty or len(prices_df) < 30:
            print("      Nedostatek cenových dat. Ukončuji.")
            sys.exit(1)
        print(f"      Staženo {len(prices_df)} dní pro {list(prices_df.columns)}")
    except Exception as e:
        print(f"      CHYBA: {e}")
        sys.exit(1)

    # 4. Markowitz optimization
    print("\n[4/5] Optimalizace portfolia (Markowitz – max Sharpe)...")
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    optimizer.set_prices(prices_df)
    weights = optimizer.optimize_sharpe_ratio()

    if not weights:
        print("      Optimalizace nevrátila váhy.")
    else:
        weight_rows = [[t, f"{w*100:.2f} %"] for t, w in sorted(weights.items(), key=lambda x: -x[1])]
        print(tabulate(weight_rows, headers=["Ticker", "Doporučený podíl"], tablefmt="rounded_grid"))
        print("\n      Doporučení: Kolik procent portfolia alokovat do každé akcie (výše).")

    # 5. Smart DCA for each selected ticker
    print("\n[5/5] Smart DCA – co dělat právě teď (Koupit / Čekat)...")
    dca = SmartDCAAnalyzer()
    for ticker in selected_tickers:
        if ticker not in prices_df.columns:
            continue
        series = prices_df[ticker].dropna()
        current_price = float(series.iloc[-1])
        action_key, message = dca.analyze_entry(ticker, current_price, series)
        print(f"      {message}")

    print("\n" + "=" * 60)
    print("  Konec analýzy")
    print("=" * 60)


if __name__ == "__main__":
    main()
