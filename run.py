"""
Conseiller Épargne Entreprise — point d'entrée principal.

Usage:
    python run.py                # analyse complète + conseil IA
    python run.py --no-ai        # analyse quantitative seulement (sans clé API)
    python run.py --ask "..."    # poser une question spécifique à l'IA

Variables d'environnement:
    ANTHROPIC_API_KEY  — clé API Anthropic (requis pour les conseils IA)
"""

import argparse
import os
import sys

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Conseiller Épargne Entreprise")
    parser.add_argument("--no-ai", action="store_true", help="Analyse quantitative seulement")
    parser.add_argument("--ask", type=str, default=None, help="Question spécifique pour l'IA")
    args = parser.parse_args()

    use_ai = not args.no_ai and bool(os.getenv("ANTHROPIC_API_KEY"))
    if not args.no_ai and not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[yellow]⚠ ANTHROPIC_API_KEY non définie — mode analyse seulement.[/yellow]")
        console.print("[dim]Définissez la variable pour activer les conseils IA Claude.[/dim]\n")

    from app.modules.reporting.reporter import (
        print_header,
        print_portfolio_summary,
        print_risk_metrics,
        print_correlation,
        print_optimizations,
        print_ai_advice,
        print_separator,
        spinner,
    )
    from app.services.risk_analyzer import analyse_portfolio, optimise_portfolio
    from app.services.market_data import get_portfolio_prices

    print_header()

    # ── Fetch market data ────────────────────────────────────────────────────
    with spinner("Récupération des données de marché (yfinance)..."):
        try:
            prices = get_portfolio_prices()
        except Exception as e:
            console.print(f"[red]Erreur données marché: {e}[/red]")
            sys.exit(1)

    # ── Portfolio analysis ───────────────────────────────────────────────────
    with spinner("Calcul des métriques de risque (PyPortfolioOpt)..."):
        analysis = analyse_portfolio()
        optimizations = optimise_portfolio(prices)

    print_portfolio_summary(analysis)
    print_separator()
    print_risk_metrics(analysis)
    print_separator()
    print_correlation(analysis)
    print_separator()
    print_optimizations(optimizations)
    print_separator()

    # ── AI advisor ───────────────────────────────────────────────────────────
    if use_ai:
        from app.services.advisor import ask_advisor, get_quick_advice
        with spinner("Analyse IA en cours (Claude Opus 4.7 + adaptive thinking)..."):
            question = args.ask if args.ask else None
            if question:
                advice = ask_advisor(question, analysis, optimizations)
            else:
                advice = get_quick_advice(analysis, optimizations)
        print_ai_advice(advice)

        # Interactive mode
        if not args.ask:
            console.print("\n[dim]Mode interactif — tapez votre question (ou 'quitter') :[/dim]")
            while True:
                try:
                    q = console.input("[cyan]Vous > [/cyan]").strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if q.lower() in ("quitter", "quit", "exit", "q"):
                    break
                if not q:
                    continue
                with spinner("Réflexion..."):
                    response = ask_advisor(q, analysis, optimizations)
                print_ai_advice(response)
    else:
        console.print("\n[dim]Mode analyse seulement. Définissez ANTHROPIC_API_KEY pour les conseils IA.[/dim]")

    console.print("\n[bold green]✓ Analyse terminée.[/bold green]")


if __name__ == "__main__":
    main()
