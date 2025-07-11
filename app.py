import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from src.optimizers import optimize_mvo, optimize_risk_parity, optimize_hrp
from src.backtest import run_backtest
from src.black_litterman import run_black_litterman
from src.utils import load_data

app = dash.Dash(__name__)
server = app.server

price_df, returns_df = load_data()
strategies = ["MVO", "Risk Parity", "HRP", "Black-Litterman"]

app.layout = html.Div([
    html.H1("Portfolio Optimization Dashboard", style={"textAlign": "center"}),
    dcc.Dropdown(id='strategy', options=[{"label": s, "value": s} for s in strategies], value='MVO'),
    dcc.Graph(id='performance-graph'),
    dcc.Graph(id='allocation-graph'),
    dcc.Graph(id='risk-graph')
])

@app.callback(
    [Output('performance-graph', 'figure'),
     Output('allocation-graph', 'figure'),
     Output('risk-graph', 'figure')],
    [Input('strategy', 'value')]
)
def update_dashboard(strategy):
    if strategy == "MVO":
        weights = optimize_mvo(returns_df, returns_df.cov())
    elif strategy == "Risk Parity":
        weights = optimize_risk_parity(returns_df)
    elif strategy == "HRP":
        weights = optimize_hrp(returns_df)
    elif strategy == "Black-Litterman":
        weights = run_black_litterman(returns_df)
    else:
        weights = None

    weights_df = pd.DataFrame(weights, index=returns_df.columns, columns=["Weight"])
    alloc_fig = go.Figure([go.Bar(x=weights_df.index, y=weights_df["Weight"])]).update_layout(title="Portfolio Allocation")

    simulated_weights = pd.DataFrame([weights_df["Weight"]]*len(price_df),
                                     index=price_df.index,
                                     columns=weights_df.index)
    cumulative, sharpe, dd = run_backtest(price_df, simulated_weights)
    perf_fig = go.Figure([go.Scatter(x=cumulative.index, y=cumulative, mode='lines')])
    perf_fig.update_layout(title=f'Cumulative Return | Sharpe: {sharpe:.2f} | Max DD: {dd:.2%}')

    rc_fig = go.Figure([go.Pie(labels=weights_df.index, values=weights_df["Weight"])])
    rc_fig.update_layout(title="Risk Contribution (Proxy)")

    return perf_fig, alloc_fig, rc_fig

if __name__ == '__main__':
    app.run_server(debug=True)
