"""Portfolio configuration — maps French PEE funds to yfinance proxy tickers."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FundConfig:
    name: str
    short_name: str
    proxy_ticker: str       # yfinance ticker (nearest public equivalent)
    value_eur: float        # current holding in euros
    shares: float           # units held
    unrealised_pnl: float   # unrealised P&L
    risk_level: int         # SRRI 1-7
    ytd_performance: Optional[float]
    asset_class: str
    zone: str


PORTFOLIO_FUNDS: list[FundConfig] = [
    FundConfig(
        name="MULTIPAR ACTIONS INDICE MONDE [L, C]",
        short_name="Monde Index",
        proxy_ticker="IWDA.AS",   # iShares MSCI World UCITS ETF
        value_eur=654.28,
        shares=61.5676,
        unrealised_pnl=22.70,
        risk_level=4,
        ytd_performance=None,
        asset_class="Actions Monde",
        zone="Monde",
    ),
    FundConfig(
        name="MULTIPAR ACTIONS SOCIALEMENT RESPONSABLE [CLASSIQUE, C]",
        short_name="ISR / ESG",
        proxy_ticker="SWRD.L",    # SPDR MSCI World UCITS ETF
        value_eur=511.66,
        shares=10.5584,
        unrealised_pnl=14.96,
        risk_level=4,
        ytd_performance=2.04,
        asset_class="Actions ISR",
        zone="Monde",
    ),
    FundConfig(
        name="BNP PARIBAS ACTIONNARIAT FRANCE [CLASSIQUE, C]",
        short_name="Actionnariat FR",
        proxy_ticker="^FCHI",     # CAC 40 index
        value_eur=220.22,
        shares=2.4116,
        unrealised_pnl=-1.43,
        risk_level=6,
        ytd_performance=15.39,
        asset_class="Actions France",
        zone="France",
    ),
]

TOTAL_VALUE = sum(f.value_eur for f in PORTFOLIO_FUNDS)
TOTAL_PNL = sum(f.unrealised_pnl for f in PORTFOLIO_FUNDS)

BENCHMARKS = {
    "MSCI World": "IWDA.AS",
    "CAC 40": "^FCHI",
}

RISK_FREE_RATE = 0.03   # taux livret A / BCE
LOOKBACK_YEARS = 3
