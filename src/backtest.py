def run_backtest(price_df, weights_df, rebalance_freq='M'):
    import pandas as pd
    import numpy as np

    returns = price_df.pct_change().dropna()
    rebalance_dates = weights_df.index
    portfolio_returns = []

    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]
        weights = weights_df.loc[start].values
        period_returns = returns.loc[start:end]
        weighted_returns = (period_returns * weights).sum(axis=1)
        portfolio_returns.append(weighted_returns)

    full_returns = pd.concat(portfolio_returns)
    cumulative = (1 + full_returns).cumprod()
    sharpe = (full_returns.mean() / full_returns.std()) * np.sqrt(252)
    drawdown = (cumulative / cumulative.cummax() - 1).min()
    return cumulative, sharpe, drawdown
