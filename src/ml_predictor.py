import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt

def load_features(tickers, start, end, lags=5):
    df = yf.download(tickers, start=start, end=end)[['Adj Close', 'Volume']]
    data = pd.DataFrame({f'{t}_{col}': df[col][t] for t in tickers for col in ['Adj Close', 'Volume']})
    for t in tickers:
        data[f'{t}_ret'] = data[f'{t}_Adj Close'].pct_change().shift(-21)
        for lag in range(1, lags+1):
            data[f'{t}_lag_ret{lag}'] = data[f'{t}_Adj Close'].pct_change(lag)
            data[f'{t}_lag_vol{lag}'] = data[f'{t}_Volume'].pct_change(lag)
    data = data.dropna()
    return data

def train_predictor(data, target_col, test_size=0.2):
    X = data.drop(columns=[target_col])
    y = data[target_col]
    split = int(len(data) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"{target_col} – Test RMSE: {np.sqrt(mse):.4f}")

    return model, X_test, preds

def compute_shap(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("shap_summary.png")
    return shap_values

def predict_all_and_rank(tickers, period_data):
    preds = {}
    for t in tickers:
        data = load_features([t], *period_data)
        model, X_test, p = train_predictor(data, f'{t}_ret')
        shap_vals = compute_shap(model, X_test)
        preds[t] = p[-1]
    ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked[:5])
