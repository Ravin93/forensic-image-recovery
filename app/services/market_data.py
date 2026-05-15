"""
Market data for fund proxies.
Tries yfinance first; falls back to simulated returns using realistic
long-run parameters when the network is unavailable.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.core.config import PORTFOLIO_FUNDS, LOOKBACK_YEARS, RISK_FREE_RATE

warnings.filterwarnings("ignore")

# ── Realistic parameters for French PEE fund categories ────────────────────
# Source: long-run MSCI / Bloomberg data (2000-2024 averages)
FUND_PARAMS = {
    "Monde Index": {
        "annual_mu": 0.099,    # MSCI World ~9.9%/year
        "annual_sigma": 0.148, # volatility ~14.8%
        "ticker": "IWDA.AS",
    },
    "ISR / ESG": {
        "annual_mu": 0.094,    # ESG World slightly lower than pure market
        "annual_sigma": 0.142,
        "ticker": "SWRD.L",
    },
    "Actionnariat FR": {
        "annual_mu": 0.072,    # CAC 40 historically lower + higher vol
        "annual_sigma": 0.198,
        "ticker": "^FCHI",
    },
}

# Realistic pairwise correlations between fund categories
CORRELATIONS = np.array([
    [1.00, 0.95, 0.75],   # Monde vs [Monde, ISR, France]
    [0.95, 1.00, 0.72],   # ISR   vs [Monde, ISR, France]
    [0.75, 0.72, 1.00],   # France vs [Monde, ISR, France]
])


def _try_yfinance(tickers: list[str], years: int) -> pd.DataFrame | None:
    """Attempt live download; return None on any failure."""
    import sys, io
    try:
        import yfinance as yf
        # Suppress yfinance error output
        _stderr = sys.stderr
        sys.stderr = io.StringIO()
        end = datetime.today()
        start = end - timedelta(days=365 * years)
        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw
        prices = prices.dropna(how="all")
        if prices.empty or prices.isnull().all().all():
            sys.stderr = _stderr
            return None
        sys.stderr = _stderr
        return prices
    except Exception:
        try:
            sys.stderr = _stderr
        except Exception:
            pass
        return None


def _simulate_prices(years: int = LOOKBACK_YEARS, seed: int = 42) -> pd.DataFrame:
    """
    Generate correlated synthetic daily prices using realistic parameters.
    Uses Cholesky decomposition so the funds are properly correlated.
    """
    rng = np.random.default_rng(seed)
    trading_days = int(years * 252)
    names = list(FUND_PARAMS.keys())
    n = len(names)

    daily_mu = np.array([FUND_PARAMS[k]["annual_mu"] / 252 for k in names])
    daily_sigma = np.array([FUND_PARAMS[k]["annual_sigma"] / np.sqrt(252) for k in names])

    L = np.linalg.cholesky(CORRELATIONS)
    z = rng.standard_normal((trading_days, n))
    correlated_z = z @ L.T

    daily_returns = daily_mu + correlated_z * daily_sigma
    prices = pd.DataFrame(
        100 * np.cumprod(1 + daily_returns, axis=0),
        columns=names,
        index=pd.bdate_range(end=datetime.today(), periods=trading_days),
    )
    return prices


def get_portfolio_prices() -> pd.DataFrame:
    """Return daily close prices for all fund proxies, named by short_name."""
    tickers = [FUND_PARAMS[f.short_name]["ticker"] for f in PORTFOLIO_FUNDS
               if f.short_name in FUND_PARAMS]
    live = _try_yfinance(tickers, LOOKBACK_YEARS)

    if live is not None and not live.empty:
        rename = {}
        for f in PORTFOLIO_FUNDS:
            p = FUND_PARAMS.get(f.short_name, {})
            rename[p.get("ticker", "")] = f.short_name
        live = live.rename(columns=rename)
        return live.dropna()

    # Fallback: realistic simulation
    return _simulate_prices(LOOKBACK_YEARS)


def get_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def compute_risk_metrics(returns: pd.Series, rf: float = RISK_FREE_RATE) -> dict:
    """Annualised risk metrics from a daily returns series."""
    if len(returns) < 5:
        # Safety guard for empty/very short series
        return {
            "annual_return": 0, "annual_volatility": 0, "sharpe_ratio": 0,
            "max_drawdown": 0, "var_95": 0, "sortino_ratio": 0,
        }
    ann = 252
    annual_return = float(returns.mean() * ann)
    annual_vol = float(returns.std() * np.sqrt(ann))

    excess = returns - rf / ann
    sharpe = float((excess.mean() / returns.std()) * np.sqrt(ann)) if returns.std() > 0 else 0.0

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    max_dd = float(((cum - rolling_max) / rolling_max).min())

    downside = returns[returns < 0]
    ds_std = downside.std() * np.sqrt(ann)
    sortino = float((annual_return - rf) / ds_std) if ds_std > 0 else 0.0

    var_95 = float(np.percentile(returns.dropna(), 5))

    return {
        "annual_return": round(annual_return, 4),
        "annual_volatility": round(annual_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "var_95": round(var_95, 4),
        "sortino_ratio": round(sortino, 3),
    }


def compute_portfolio_metrics(
    returns: pd.DataFrame, weights: list[float], rf: float = RISK_FREE_RATE
) -> dict:
    w = pd.Series(weights, index=returns.columns)
    port_returns = (returns * w).sum(axis=1)
    return compute_risk_metrics(port_returns, rf)
