def run_black_litterman(returns):
    from pypfopt import BlackLittermanModel, risk_models
    import numpy as np
    from pypfopt.efficient_frontier import EfficientFrontier

    S = risk_models.sample_cov(returns)
    mkt_weights = np.repeat(1 / returns.shape[1], returns.shape[1])
    tau = 0.05
    P = np.eye(len(mkt_weights))[:3]
    Q = np.array([0.02, -0.01, 0.03])

    bl = BlackLittermanModel(S, pi=None, absolute_views=True, P=P, Q=Q, tau=tau)
    ret_bl = bl.bl_returns()
    cov_bl = bl.bl_cov()

    ef = EfficientFrontier(ret_bl, cov_bl)
    weights = ef.max_sharpe()
    return ef.clean_weights()
