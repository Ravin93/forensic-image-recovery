"""AI financial advisor powered by Claude API with tool use."""

import json
import os

import anthropic

from app.core.config import PORTFOLIO_FUNDS, RISK_FREE_RATE, TOTAL_VALUE, TOTAL_PNL
from app.models.portfolio import PortfolioAnalysis, OptimizationResult

# ── Claude client ────────────────────────────────────────────────────────────
_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """Tu es un conseiller financier expert spécialisé dans l'épargne salariale française (PEE/PERCO).
Tu analyses des portefeuilles d'épargne entreprise et tu donnes des conseils concrets, prudents, et adaptés au profil de l'investisseur.

Règles absolues :
- Toujours prioriser la protection du capital pour les petits portefeuilles (<5000€)
- Mentionner l'abondement employeur en premier si pertinent (argent gratuit)
- Distinguer clairement conseil pédagogique et décision personnelle de l'utilisateur
- Expliquer les risques clairement sans jargon
- Donner des recommandations actionnables et concrètes
- Rappeler les cas de déblocage anticipé du PEE quand pertinent

Contexte marché : Tu analyses un PEE (Plan d'Épargne Entreprise) français avec 3 fonds.
Les fonds sont des OPCVM, pas des ETF cotés directement. Les proxies boursiers sont utilisés pour l'analyse quantitative."""


# ── Tool definitions ─────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_portfolio_summary",
        "description": "Retourne le résumé complet du portefeuille: fonds, valeurs, allocations et P&L.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_risk_analysis",
        "description": "Retourne les métriques de risque calculées: volatilité, Sharpe ratio, VaR, drawdown maximal.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_optimization_results",
        "description": "Retourne les allocations optimales calculées par PyPortfolioOpt (Max Sharpe, Min Vol, Risque Modéré).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_rebalancing_plan",
        "description": "Calcule combien d'euros déplacer entre fonds pour atteindre une allocation cible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_weights": {
                    "type": "object",
                    "description": "Poids cible par fonds court-nom (ex: {'Monde Index': 0.70, 'ISR / ESG': 0.20, 'Actionnariat FR': 0.10})",
                }
            },
            "required": ["target_weights"],
        },
    },
]


# ── Tool handlers ────────────────────────────────────────────────────────────
def _handle_tool(
    name: str,
    tool_input: dict,
    analysis: PortfolioAnalysis,
    optimizations: list[OptimizationResult],
) -> str:
    if name == "get_portfolio_summary":
        rows = []
        for f in analysis.funds:
            rows.append({
                "fonds": f.short_name,
                "valeur_eur": f.value_eur,
                "poids_pct": round(f.weight * 100, 1),
                "pnl_eur": f.unrealised_pnl,
                "risque_srri": f.risk_level,
                "ytd_pct": f.ytd_performance,
            })
        return json.dumps({
            "total_valeur_eur": analysis.total_value,
            "total_pnl_eur": round(analysis.total_pnl, 2),
            "total_pnl_pct": round(analysis.total_pnl_pct * 100, 2),
            "fonds": rows,
        }, ensure_ascii=False)

    if name == "get_risk_analysis":
        m = analysis.portfolio_metrics
        fund_metrics = []
        for f in analysis.funds:
            if f.metrics:
                fund_metrics.append({
                    "fonds": f.short_name,
                    "rendement_annuel_pct": round(f.metrics.annual_return * 100, 2),
                    "volatilite_annuelle_pct": round(f.metrics.annual_volatility * 100, 2),
                    "sharpe_ratio": f.metrics.sharpe_ratio,
                    "drawdown_max_pct": round(f.metrics.max_drawdown * 100, 2),
                    "var_95_journalier_pct": round(f.metrics.var_95 * 100, 2),
                })
        corr = {}
        if analysis.correlation_matrix is not None:
            corr = analysis.correlation_matrix.round(3).to_dict()
        return json.dumps({
            "portefeuille_global": {
                "rendement_annuel_pct": round(m.annual_return * 100, 2) if m else None,
                "volatilite_annuelle_pct": round(m.annual_volatility * 100, 2) if m else None,
                "sharpe_ratio": m.sharpe_ratio if m else None,
                "drawdown_max_pct": round(m.max_drawdown * 100, 2) if m else None,
                "var_95_journalier_pct": round(m.var_95 * 100, 2) if m else None,
            },
            "fonds_individuels": fund_metrics,
            "correlation_matrix": corr,
        }, ensure_ascii=False)

    if name == "get_optimization_results":
        out = []
        for opt in optimizations:
            out.append({
                "strategie": opt.strategy,
                "poids": {k: round(v * 100, 1) for k, v in opt.weights.items() if v > 0.01},
                "rendement_attendu_pct": round(opt.expected_return * 100, 2),
                "volatilite_attendue_pct": round(opt.expected_volatility * 100, 2),
                "sharpe_ratio": opt.sharpe_ratio,
                "description": opt.description,
            })
        return json.dumps({"optimisations": out}, ensure_ascii=False)

    if name == "get_rebalancing_plan":
        target = tool_input.get("target_weights", {})
        plan = []
        for f in analysis.funds:
            current_pct = f.weight * 100
            target_pct = target.get(f.short_name, 0) * 100
            diff_eur = (target_pct - current_pct) / 100 * analysis.total_value
            plan.append({
                "fonds": f.short_name,
                "actuel_pct": round(current_pct, 1),
                "cible_pct": round(target_pct, 1),
                "action_eur": round(diff_eur, 2),
                "action": "Acheter" if diff_eur > 0 else ("Vendre" if diff_eur < 0 else "Conserver"),
            })
        return json.dumps({"plan_reequilibrage": plan}, ensure_ascii=False)

    return json.dumps({"error": f"Outil inconnu: {name}"})


# ── Main advisor function ────────────────────────────────────────────────────
def ask_advisor(
    question: str,
    analysis: PortfolioAnalysis,
    optimizations: list[OptimizationResult],
    stream_callback=None,
) -> str:
    """
    Ask the Claude advisor a question about the portfolio.
    Returns the full text response. Calls stream_callback(chunk) if provided.
    """
    messages = [{"role": "user", "content": question}]
    full_response = ""

    while True:
        with _client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=TOOLS,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        # Collect text output
        text_blocks = [b.text for b in response.content if b.type == "text"]
        chunk = " ".join(text_blocks)
        if chunk:
            full_response += chunk
            if stream_callback:
                stream_callback(chunk)

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tu in tool_uses:
                result = _handle_tool(tu.name, tu.input, analysis, optimizations)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    return full_response


def get_quick_advice(analysis: PortfolioAnalysis, optimizations: list[OptimizationResult]) -> str:
    """Generate an initial diagnostic without user input."""
    return ask_advisor(
        "Analyse mon portefeuille d'épargne entreprise. "
        "Utilise tous les outils disponibles pour : "
        "1) évaluer la situation actuelle, "
        "2) identifier les problèmes principaux (risque, diversification, performance), "
        "3) donner 3 recommandations concrètes prioritaires. "
        "Sois précis avec les chiffres.",
        analysis,
        optimizations,
    )
