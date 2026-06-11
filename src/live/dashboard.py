"""
live/dashboard.py — Real-time performance dashboard for CARMS paper trading.

Runs a Dash web app at http://localhost:8050 showing:
  - Live portfolio value and return
  - Equity curve with regime background colours
  - Current regime probabilities
  - Per-asset positions
  - Trade log table
  - Sharpe, drawdown, win rate

Usage:
    python main.py --phase 5 --dashboard
    Open http://localhost:8050 in your browser
"""

from pathlib import Path
from typing import Optional
import json

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

REGIME_COLOURS = {
    "trending_up":   "#1D9E75",
    "trending_down": "#E24B4A",
    "ranging":       "#888780",
    "crisis":        "#D85A30",
}


def run_dashboard(save_dir: str = "models", port: int = 8050):
    try:
        import dash
        from dash import dcc, html, Input, Output
        import plotly.graph_objects as go
    except ImportError:
        log.error("Run: pip install dash plotly")
        return

    save_path = Path(save_dir)
    app = dash.Dash(__name__, title="CARMS Dashboard")

    app.layout = html.Div([
        html.Div([
            html.H2("CARMS — Cross-Asset Regime-Aware Trading System",
                    style={"color": "#1a1a2e", "margin": "0"}),
            html.P("Paper Trading Dashboard | Updates every 10s",
                   style={"color": "#888", "margin": "4px 0 0"}),
        ], style={"padding": "20px 30px 10px", "borderBottom": "1px solid #eee",
                  "background": "#fff"}),

        html.Div(id="metrics-row", style={
            "display": "flex", "gap": "16px", "padding": "16px 30px",
            "flexWrap": "wrap", "background": "#fafafa",
        }),

        html.Div([
            html.Div([dcc.Graph(id="equity-curve",    style={"height": "320px"})],
                     style={"flex": "2", "minWidth": "420px"}),
            html.Div([dcc.Graph(id="regime-bar",      style={"height": "320px"})],
                     style={"flex": "1", "minWidth": "260px"}),
        ], style={"display": "flex", "gap": "16px", "padding": "0 30px", "marginTop": "8px"}),

        html.Div([
            html.Div([dcc.Graph(id="positions-chart", style={"height": "280px"})],
                     style={"flex": "1", "minWidth": "300px"}),
            html.Div([dcc.Graph(id="regime-pie",      style={"height": "280px"})],
                     style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "16px", "padding": "0 30px", "marginTop": "16px"}),

        html.Div([
            html.H4("Recent Trades", style={"color": "#444", "marginBottom": "8px"}),
            html.Div(id="trade-table"),
        ], style={"padding": "16px 30px"}),

        dcc.Interval(id="interval", interval=10_000, n_intervals=0),
    ], style={"fontFamily": "system-ui, sans-serif", "background": "#fafafa", "minHeight": "100vh"})

    @app.callback(
        Output("metrics-row",    "children"),
        Output("equity-curve",   "figure"),
        Output("regime-bar",     "figure"),
        Output("positions-chart","figure"),
        Output("regime-pie",     "figure"),
        Output("trade-table",    "children"),
        Input("interval", "n_intervals"),
    )
    def update(_n):
        state  = _load_state(save_path)
        trades = _load_trades(save_path)
        sim    = _load_sim(save_path)
        return (
            _metrics_row(state),
            _equity_fig(sim),
            _regime_bar(sim),
            _positions_fig(state),
            _regime_pie(sim),
            _trade_table(trades),
        )

    log.info(f"Dashboard → http://localhost:{port}")
    app.run(debug=False, port=port, host="0.0.0.0")


# ── Data loaders ──────────────────────────────────────────────

def _load_state(p: Path) -> dict:
    path = p / "portfolio_state.json"
    if not path.exists(): return {}
    try:
        with open(path) as f: return json.load(f)
    except Exception: return {}

def _load_trades(p: Path) -> Optional[pd.DataFrame]:
    path = p / "trade_log.csv"
    if not path.exists(): return None
    try: return pd.read_csv(path, parse_dates=["timestamp"])
    except Exception: return None

def _load_sim(p: Path) -> Optional[pd.DataFrame]:
    path = p / "paper_trade_simulation.csv"
    if not path.exists(): return None
    try: return pd.read_csv(path, parse_dates=["date"])
    except Exception: return None


# ── Chart builders ────────────────────────────────────────────

def _metrics_row(state: dict) -> list:
    from dash import html
    val  = state.get("portfolio_value", 10_000)
    ret  = (val / 10_000 - 1) * 100
    dd   = state.get("current_drawdown_pct", 0)
    n_tr = state.get("n_trades", 0)
    halt = state.get("halted", False)

    def card(title, value, colour="#1a1a2e"):
        return html.Div([
            html.P(title, style={"margin":"0","fontSize":"12px","color":"#888"}),
            html.H3(value, style={"margin":"4px 0 0","color":colour,"fontSize":"22px"}),
        ], style={"background":"#fff","borderRadius":"10px","padding":"14px 20px",
                  "boxShadow":"0 1px 4px rgba(0,0,0,.07)","minWidth":"140px"})

    return [
        card("Portfolio",    f"${val:,.2f}"),
        card("Return",       f"{ret:+.2f}%", "#1D9E75" if ret>=0 else "#E24B4A"),
        card("Drawdown",     f"{dd:.1f}%",   "#E24B4A" if dd>10 else "#888"),
        card("Trades",       str(n_tr)),
        card("Status",       "⚠ HALTED" if halt else "● ACTIVE",
             "#E24B4A" if halt else "#1D9E75"),
    ]


def _equity_fig(sim: Optional[pd.DataFrame]):
    import plotly.graph_objects as go
    fig = go.Figure()
    if sim is not None and "portfolio_value" in sim.columns:
        x = sim["date"] if "date" in sim.columns else sim.index
        fig.add_trace(go.Scatter(
            x=x, y=sim["portfolio_value"], name="Portfolio",
            line=dict(color="#534AB7", width=2),
            fill="tozeroy", fillcolor="rgba(83,74,183,0.07)",
        ))
        fig.add_hline(y=10_000, line_dash="dot", line_color="#ccc",
                      annotation_text="Initial $10,000")
    fig.update_layout(title="Portfolio Equity Curve", xaxis_title="Date",
                      yaxis_title="Value ($)", paper_bgcolor="white",
                      plot_bgcolor="#f8f9fa", margin=dict(l=40,r=20,t=40,b=40))
    return fig


def _regime_bar(sim: Optional[pd.DataFrame]):
    import plotly.graph_objects as go
    names = list(REGIME_COLOURS.keys())
    probs = [0.25, 0.25, 0.25, 0.25]
    if sim is not None and "regime" in sim.columns:
        counts = sim["regime"].value_counts(normalize=True)
        probs  = [float(counts.get(n, 0)) for n in names]
    fig = go.Figure(go.Bar(
        x=[n.replace("_"," ").title() for n in names],
        y=probs,
        marker_color=list(REGIME_COLOURS.values()),
        text=[f"{p:.0%}" for p in probs], textposition="outside",
    ))
    latest_regime = sim["regime"].iloc[-1] if sim is not None and "regime" in sim.columns else "?"
    fig.update_layout(
        title=f"Regime Frequency  (latest: {latest_regime.replace('_',' ').title()})",
        yaxis=dict(range=[0,1.1], tickformat=".0%"),
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        margin=dict(l=40,r=20,t=40,b=40),
    )
    return fig


def _positions_fig(state: dict):
    import plotly.graph_objects as go
    positions = state.get("positions", {})
    if not positions:
        fig = go.Figure()
        fig.add_annotation(text="No open positions", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
    else:
        syms = list(positions.keys())
        qtys = [positions[s] for s in syms]
        fig = go.Figure(go.Bar(
            x=syms, y=qtys,
            marker_color=["#1D9E75" if q>0 else "#E24B4A" for q in qtys],
            text=[f"{q:+.4f}" for q in qtys], textposition="outside",
        ))
    fig.update_layout(title="Open Positions", yaxis_title="Quantity",
                      paper_bgcolor="white", plot_bgcolor="#f8f9fa",
                      margin=dict(l=40,r=20,t=40,b=40))
    return fig


def _regime_pie(sim: Optional[pd.DataFrame]):
    import plotly.graph_objects as go
    if sim is None or "regime" not in sim.columns:
        fig = go.Figure()
        fig.add_annotation(text="Run simulation first", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title="Regime Distribution", paper_bgcolor="white",
                          margin=dict(l=20,r=20,t=40,b=20))
        return fig
    counts = sim["regime"].value_counts()
    fig = go.Figure(go.Pie(
        labels=[r.replace("_"," ").title() for r in counts.index],
        values=counts.values,
        marker_colors=[REGIME_COLOURS.get(r,"#888") for r in counts.index],
        hole=0.4,
    ))
    fig.update_layout(title="Regime Distribution", paper_bgcolor="white",
                      margin=dict(l=20,r=20,t=40,b=20))
    return fig


def _trade_table(trades: Optional[pd.DataFrame]):
    from dash import html
    if trades is None or trades.empty:
        return html.P("No trades yet.", style={"color":"#888"})
    recent = trades.tail(10).iloc[::-1]
    th = lambda t: html.Th(t, style={"padding":"8px 12px","textAlign":"left",
                                      "fontSize":"12px","color":"#888",
                                      "borderBottom":"2px solid #eee"})
    header = html.Tr([th(c) for c in ["Time","Symbol","Direction","Qty","Price","Value","Portfolio"]])
    rows = []
    for _, row in recent.iterrows():
        d = row.get("direction","")
        col = "#1D9E75" if d=="LONG" else ("#E24B4A" if d=="SHORT" else "#888")
        td = lambda v, **s: html.Td(v, style={"padding":"6px 12px","fontSize":"12px",**s})
        rows.append(html.Tr([
            td(str(row.get("timestamp",""))[:16]),
            td(str(row.get("symbol","")),  fontWeight="500"),
            td(d, color=col, fontWeight="bold"),
            td(f"{row.get('quantity',0):+.4f}"),
            td(f"${row.get('price',0):,.4f}"),
            td(f"${row.get('trade_value',0):,.2f}"),
            td(f"${row.get('portfolio_val',0):,.2f}", fontWeight="500"),
        ], style={"borderBottom":"1px solid #f0f0f0"}))
    return html.Table([header]+rows, style={
        "width":"100%","borderCollapse":"collapse",
        "background":"#fff","borderRadius":"8px",
        "boxShadow":"0 1px 4px rgba(0,0,0,.07)",
    })