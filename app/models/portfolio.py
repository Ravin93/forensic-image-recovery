"""Data models for portfolio analysis."""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class RiskMetrics:
    annual_return: float        # expected annual return
    annual_volatility: float    # annual standard deviation
    sharpe_ratio: float         # Sharpe ratio
    max_drawdown: float         # maximum drawdown (negative)
    var_95: float               # Value at Risk 95% (1-day, negative)
    sortino_ratio: float        # Sortino ratio


@dataclass
class FundAnalysis:
    short_name: str
    ticker: str
    value_eur: float
    weight: float               # current weight in portfolio
    unrealised_pnl: float
    risk_level: int
    ytd_performance: Optional[float]
    metrics: Optional[RiskMetrics] = None
    returns_series: Optional[pd.Series] = None


@dataclass
class PortfolioAnalysis:
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    funds: list[FundAnalysis]
    portfolio_metrics: Optional[RiskMetrics] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    optimal_weights: Optional[dict[str, float]] = None
    rebalancing_suggestions: list[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    strategy: str               # e.g. "Max Sharpe", "Min Volatility"
    weights: dict[str, float]   # fund short_name -> weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    description: str
