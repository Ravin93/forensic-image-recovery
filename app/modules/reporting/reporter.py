"""Rich console reporting for portfolio analysis."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
from contextlib import contextmanager

from app.models.portfolio import PortfolioAnalysis, OptimizationResult

console = Console()


@contextmanager
def spinner(message: str):
    with Progress(SpinnerColumn(), TextColumn(message), transient=True) as p:
        p.add_task("", total=None)
        yield


def _pnl_color(val: float) -> str:
    return "green" if val >= 0 else "red"


def print_header():
    console.print(Panel(
        "[bold cyan]Conseiller Épargne Entreprise[/bold cyan]\n"
        "[dim]Propulsé par PyPortfolioOpt + Claude Opus 4.7[/dim]",
        box=box.DOUBLE,
        style="cyan",
    ))


def print_portfolio_summary(analysis: PortfolioAnalysis):
    pnl_color = _pnl_color(analysis.total_pnl)
    pnl_str = f"[{pnl_color}]{analysis.total_pnl:+.2f} € ({analysis.total_pnl_pct*100:+.2f}%)[/{pnl_color}]"

    table = Table(
        title=f"[bold]Portefeuille total : [cyan]{analysis.total_value:.2f} €[/cyan]  |  P&L : {pnl_str}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Fonds", style="bold white", min_width=20)
    table.add_column("Valeur (€)", justify="right", style="cyan")
    table.add_column("Poids", justify="right")
    table.add_column("P&L (€)", justify="right")
    table.add_column("Risque SRRI", justify="center")
    table.add_column("Perf. YTD", justify="right")

    risk_colors = {1: "green", 2: "green", 3: "yellow", 4: "yellow", 5: "orange3", 6: "red", 7: "red1"}

    for f in analysis.funds:
        risk_col = risk_colors.get(f.risk_level, "white")
        pnl_c = _pnl_color(f.unrealised_pnl)
        ytd = f"{f.ytd_performance:+.2f}%" if f.ytd_performance is not None else "—"
        table.add_row(
            f.short_name,
            f"{f.value_eur:.2f}",
            f"{f.weight*100:.1f}%",
            f"[{pnl_c}]{f.unrealised_pnl:+.2f}[/{pnl_c}]",
            f"[{risk_col}]{f.risk_level}/7[/{risk_col}]",
            ytd,
        )

    console.print(table)


def print_risk_metrics(analysis: PortfolioAnalysis):
    m = analysis.portfolio_metrics
    if not m:
        return

    table = Table(title="[bold]Métriques de risque (données historiques 3 ans)[/bold]", box=box.SIMPLE_HEAVY)
    table.add_column("Indicateur", style="bold")
    table.add_column("Portefeuille", justify="right")
    for f in analysis.funds:
        table.add_column(f.short_name, justify="right")

    def _fmt_pct(v): return f"{v*100:.2f}%"
    def _fmt_ratio(v): return f"{v:.3f}"

    rows = [
        ("Rendement annuel", _fmt_pct(m.annual_return),
         [_fmt_pct(f.metrics.annual_return) if f.metrics else "—" for f in analysis.funds]),
        ("Volatilité annuelle", _fmt_pct(m.annual_volatility),
         [_fmt_pct(f.metrics.annual_volatility) if f.metrics else "—" for f in analysis.funds]),
        ("Sharpe ratio", _fmt_ratio(m.sharpe_ratio),
         [_fmt_ratio(f.metrics.sharpe_ratio) if f.metrics else "—" for f in analysis.funds]),
        ("Drawdown max.", f"[red]{_fmt_pct(m.max_drawdown)}[/red]",
         [f"[red]{_fmt_pct(f.metrics.max_drawdown)}[/red]" if f.metrics else "—" for f in analysis.funds]),
        ("VaR 95% (jour)", f"[orange3]{_fmt_pct(m.var_95)}[/orange3]",
         [f"[orange3]{_fmt_pct(f.metrics.var_95)}[/orange3]" if f.metrics else "—" for f in analysis.funds]),
        ("Sortino ratio", _fmt_ratio(m.sortino_ratio),
         [_fmt_ratio(f.metrics.sortino_ratio) if f.metrics else "—" for f in analysis.funds]),
    ]

    for label, port_val, fund_vals in rows:
        table.add_row(label, port_val, *fund_vals)

    console.print(table)


def print_correlation(analysis: PortfolioAnalysis):
    if analysis.correlation_matrix is None:
        return
    corr = analysis.correlation_matrix
    console.print("\n[bold]Corrélations entre fonds[/bold] [dim](1.0 = identique, 0 = indépendant)[/dim]")
    table = Table(box=box.MINIMAL)
    table.add_column("")
    for col in corr.columns:
        table.add_column(col, justify="right")
    for idx, row in corr.iterrows():
        vals = []
        for v in row:
            if v >= 0.9:
                vals.append(f"[red]{v:.2f}[/red]")
            elif v >= 0.7:
                vals.append(f"[orange3]{v:.2f}[/orange3]")
            else:
                vals.append(f"[green]{v:.2f}[/green]")
        table.add_row(str(idx), *vals)
    console.print(table)


def print_optimizations(optimizations: list[OptimizationResult]):
    console.print("\n[bold cyan]Allocations optimales calculées (PyPortfolioOpt)[/bold cyan]")
    for opt in optimizations:
        panel_lines = [
            f"[bold]{opt.strategy}[/bold]",
            f"[dim]{opt.description}[/dim]",
            "",
        ]
        for name, w in opt.weights.items():
            if w > 0.01:
                bar = "█" * int(w * 30)
                panel_lines.append(f"  {name:<22} {w*100:5.1f}%  {bar}")
        panel_lines += [
            "",
            f"Rendement attendu : [cyan]{opt.expected_return*100:.2f}%/an[/cyan]",
            f"Volatilité attendue : [yellow]{opt.expected_volatility*100:.2f}%[/yellow]",
            f"Sharpe ratio : [green]{opt.sharpe_ratio:.3f}[/green]",
        ]
        color = "green" if "Sharpe" in opt.strategy else ("blue" if "Modéré" in opt.strategy else "yellow")
        console.print(Panel("\n".join(panel_lines), border_style=color, padding=(0, 1)))


def print_ai_advice(advice: str):
    console.print(Panel(
        advice,
        title="[bold magenta]Analyse & Recommandations — Claude Opus 4.7[/bold magenta]",
        border_style="magenta",
        padding=(1, 2),
    ))


def print_separator():
    console.print()
