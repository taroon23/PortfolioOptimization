# Multi-Strategy Portfolio Optimizer

A comprehensive portfolio optimization and asset management dashboard leveraging:
- Modern Portfolio Theory (MPT)
- Risk Parity (RP)
- Hierarchical Risk Parity (HRP)
- Black-Litterman Model (BL)
- Full backtesting suite with performance metrics

Built using Python, Plotly Dash, and Riskfolio-Lib.

## Features

- Multiple strategy-based portfolio optimizations
- Asset class constraints (Equities, Bonds, ETFs, Commodities)
- Portfolio analytics: Sharpe, Sortino, Max Drawdown, VaR
- Dynamic dashboards for allocation, risk, performance
- Black-Litterman support for user-defined market views
- Monthly/Quarterly rebalancing with backtesting

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
portfolio_optimizer/
├── data/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── optimizers.py
│   ├── backtest.py
│   ├── dashboard.py
│   └── black_litterman.py
├── app.py
└── README.md
```
