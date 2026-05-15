"""Risk analysis and portfolio optimisation using PyPortfolioOpt."""

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation

from app.core.config import PORTFOLIO_FUNDS, RISK_FREE_RATE, TOTAL_VALUE
from app.models.portfolio import FundAnalysis, OptimizationResult, PortfolioAnalysis, RiskMetrics
from app.services.market_data import (
    compute_portfolio_metrics,
    compute_risk_metrics,
    get_daily_returns,
    get_portfolio_prices,
)


def _make_risk_metrics(m: dict) -> RiskMetrics:
    return RiskMetrics(
        annual_return=m["annual_return"],
        annual_volatility=m["annual_volatility"],
        sharpe_ratio=m["sharpe_ratio"],
        max_drawdown=m["max_drawdown"],
        var_95=m["var_95"],
        sortino_ratio=m["sortino_ratio"],
    )


def analyse_portfolio() -> PortfolioAnalysis:
    """Full portfolio analysis: individual funds + aggregate metrics."""
    prices = get_portfolio_prices()
    returns = get_daily_returns(prices)

    funds: list[FundAnalysis] = []
    for fund in PORTFOLIO_FUNDS:
        sn = fund.short_name
        if sn not in returns.columns:
            continue
        m = compute_risk_metrics(returns[sn])
        rm = _make_risk_metrics(m)
        funds.append(
            FundAnalysis(
                short_name=sn,
                ticker=fund.proxy_ticker,
                value_eur=fund.value_eur,
                weight=fund.value_eur / TOTAL_VALUE,
                unrealised_pnl=fund.unrealised_pnl,
                risk_level=fund.risk_level,
                ytd_performance=fund.ytd_performance,
                metrics=rm,
                returns_series=returns[sn],
            )
        )

    weights = [f.weight for f in funds]
    port_m = compute_portfolio_metrics(returns, weights)
    port_rm = _make_risk_metrics(port_m)

    corr = returns.corr()
    total_cost = sum(f.value_eur for f in PORTFOLIO_FUNDS)
    total_pnl = sum(f.unrealised_pnl for f in PORTFOLIO_FUNDS)

    return PortfolioAnalysis(
        total_value=total_cost,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl / (total_cost - total_pnl) if (total_cost - total_pnl) else 0,
        funds=funds,
        portfolio_metrics=port_rm,
        correlation_matrix=corr,
    )


def optimise_portfolio(prices: pd.DataFrame) -> list[OptimizationResult]:
    """Run multiple optimisation strategies and return results."""
    from app.services.market_data import FUND_PARAMS

    # Use long-run expected returns directly rather than short-window estimates
    # so that optimisation is not distorted by simulation noise.
    mu_values = {k: FUND_PARAMS[k]["annual_mu"] for k in prices.columns if k in FUND_PARAMS}
    mu = pd.Series(mu_values)

    # Covariance from the simulated price series (captures correlation structure)
    S = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()

    results: list[OptimizationResult] = []

    # ── 1. Maximum Sharpe ───────────────────────────────────────────────────
    try:
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        w_raw = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
        results.append(
            OptimizationResult(
                strategy="Max Sharpe (meilleur rendement/risque)",
                weights=dict(w_raw),
                expected_return=round(perf[0], 4),
                expected_volatility=round(perf[1], 4),
                sharpe_ratio=round(perf[2], 3),
                description="Maximise le ratio rendement/risque — recommandé horizon 5+ ans.",
            )
        )
    except Exception:
        pass

    # ── 2. Minimum Volatility ───────────────────────────────────────────────
    try:
        ef2 = EfficientFrontier(mu, S)
        ef2.min_volatility()
        w2 = ef2.clean_weights()
        p2 = ef2.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
        results.append(
            OptimizationResult(
                strategy="Min Volatilité (sécurité maximale)",
                weights=dict(w2),
                expected_return=round(p2[0], 4),
                expected_volatility=round(p2[1], 4),
                sharpe_ratio=round(p2[2], 3),
                description="Minimise les fluctuations — recommandé horizon < 3 ans.",
            )
        )
    except Exception:
        pass

    # ── 3. Efficient Risk (target 10% vol) ──────────────────────────────────
    try:
        ef3 = EfficientFrontier(mu, S)
        ef3.efficient_risk(target_volatility=0.10)
        w3 = ef3.clean_weights()
        p3 = ef3.portfolio_performance(risk_free_rate=RISK_FREE_RATE, verbose=False)
        results.append(
            OptimizationResult(
                strategy="Risque Modéré (cible 10% vol.)",
                weights=dict(w3),
                expected_return=round(p3[0], 4),
                expected_volatility=round(p3[1], 4),
                sharpe_ratio=round(p3[2], 3),
                description="Equilibre rendement/risque avec volatilité plafonnée à 10%.",
            )
        )
    except Exception:
        pass

    return results


def compute_discrete_allocation(
    weights: dict[str, float],
    prices: pd.DataFrame,
    total_eur: float = TOTAL_VALUE,
) -> dict:
    """Convert continuous weights into actionable euro amounts."""
    latest = prices.iloc[-1]
    alloc = {}
    for name, w in weights.items():
        if w > 0.01:
            alloc[name] = round(w * total_eur, 2)
    return alloc
