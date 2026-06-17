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
    "trending_up":   "#00e676",  # Neon Green
    "trending_down": "#ff3860",  # Neon Red
    "ranging":       "#7b8a97",  # Steel Blue/Grey
    "crisis":        "#ff9f43",  # Amber/Orange
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
    
    external_stylesheets = [
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap"
    ]
    app = dash.Dash(__name__, title="CARMS Terminal", external_stylesheets=external_stylesheets)

    app.layout = html.Div([


        # Header
        html.Div([
            html.Div([
                html.H2("CARMS // REGIME-AWARE TERMINAL",
                        style={"color": "#ffffff", "margin": "0", "fontFamily": "'Outfit', sans-serif", "fontWeight": "700", "letterSpacing": "1px"}),
                html.Div([
                    html.Span("●", style={"color": "#00e676", "marginRight": "6px", "animation": "blink 1.5s infinite"}),
                    html.Span("LIVE FEED ACTIVE | UPDATING EVERY 10S", style={"color": "#8892b0", "fontSize": "11px", "fontFamily": "'JetBrains Mono', monospace", "letterSpacing": "0.5px"}),
                ], style={"display": "flex", "alignItems": "center", "marginTop": "4px"}),
            ]),
            html.Div([
                html.Div("STAGING PAPER-TRADING", style={
                    "border": "1px solid #ff9f43", "color": "#ff9f43", "padding": "4px 10px",
                    "borderRadius": "4px", "fontSize": "11px", "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "bold", "letterSpacing": "1px"
                })
            ])
        ], style={
            "padding": "20px 30px", "borderBottom": "1px solid #1f293d",
            "background": "#0b0e1a", "display": "flex", "justifyContent": "space-between", "alignItems": "center"
        }),

        # Metrics Row
        html.Div(id="metrics-row", style={
            "display": "flex", "gap": "16px", "padding": "24px 30px 16px",
            "flexWrap": "wrap", "background": "#080b11",
        }),

        # Charts Section Row 1
        html.Div([
            html.Div([dcc.Graph(id="equity-curve",    style={"height": "340px"})],
                     style={"flex": "2", "minWidth": "420px", "background": "#111625", "border": "1px solid #1f293d", "borderRadius": "6px", "padding": "10px"}),
            html.Div([dcc.Graph(id="regime-bar",      style={"height": "340px"})],
                     style={"flex": "1", "minWidth": "280px", "background": "#111625", "border": "1px solid #1f293d", "borderRadius": "6px", "padding": "10px"}),
        ], style={"display": "flex", "gap": "16px", "padding": "0 30px", "marginTop": "8px"}),

        # Charts Section Row 2
        html.Div([
            html.Div([dcc.Graph(id="positions-chart", style={"height": "300px"})],
                     style={"flex": "1", "minWidth": "300px", "background": "#111625", "border": "1px solid #1f293d", "borderRadius": "6px", "padding": "10px"}),
            html.Div([dcc.Graph(id="regime-pie",      style={"height": "300px"})],
                     style={"flex": "1", "minWidth": "300px", "background": "#111625", "border": "1px solid #1f293d", "borderRadius": "6px", "padding": "10px"}),
        ], style={"display": "flex", "gap": "16px", "padding": "0 30px", "marginTop": "16px"}),

        # Trade Log Table
        html.Div([
            html.H4("TRANSACTION & ORDER ACTIVITY LOG", style={
                "color": "#ffffff", "fontFamily": "'Outfit', sans-serif", 
                "marginBottom": "12px", "letterSpacing": "1px", "fontSize": "13px", "fontWeight": "600"
            }),
            html.Div(id="trade-table"),
        ], style={"padding": "24px 30px 40px"}),

        dcc.Interval(id="interval", interval=10_000, n_intervals=0),
    ], style={"fontFamily": "'Outfit', sans-serif", "background": "#080b11", "minHeight": "100vh", "color": "#ffffff"})

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

    def card(title, value, colour="#ffffff", subtext=None, subtext_color="#8892b0"):
        return html.Div([
            html.P(title, style={"margin":"0","fontSize":"11px","color":"#8892b0","fontWeight":"600","textTransform":"uppercase","letterSpacing":"1px","fontFamily":"'Outfit', sans-serif"}),
            html.H3(value, style={"margin":"8px 0 2px","color":colour,"fontSize":"26px","fontWeight":"700","fontFamily":"'JetBrains Mono', monospace"}),
            html.P(subtext, style={"margin":"0","fontSize":"11px","color":subtext_color,"fontFamily":"'Outfit', sans-serif"}) if subtext else None
        ], className="trading-card", style={"flex":"1","minWidth":"160px"})

    status_color = "#ff3860" if halt else "#00e676"
    status_text = "HALTED" if halt else "ACTIVE"

    return [
        card("Portfolio Value", f"${val:,.2f}", "#ffffff", "Initial: $10,000.00"),
        card("Total Return", f"{ret:+.2f}%", "#00e676" if ret>=0 else "#ff3860", "Net profit/loss"),
        card("Max Drawdown", f"{dd:.2f}%", "#ff3860" if dd>5 else "#8892b0", "Peak-to-trough risk"),
        card("Executed Trades", str(n_tr), "#3b82f6", f"{n_tr} operations total"),
        card("System Status", status_text, status_color, "Auto-safety check"),
    ]


def _equity_fig(sim: Optional[pd.DataFrame]):
    import plotly.graph_objects as go
    fig = go.Figure()
    if sim is not None and "portfolio_value" in sim.columns:
        x = sim["date"] if "date" in sim.columns else sim.index
        fig.add_trace(go.Scatter(
            x=x, y=sim["portfolio_value"], name="Portfolio Value",
            line=dict(color="#3b82f6", width=2.5),
            fill="tozeroy", 
            fillcolor="rgba(59,130,246,0.06)",
            hoverinfo="x+y"
        ))
        fig.add_hline(y=10_000, line_dash="dash", line_color="#8892b0", line_width=1,
                      annotation_text="Initial Capital ($10,000)", annotation_position="bottom left",
                      annotation_font=dict(color="#8892b0", size=10, family="JetBrains Mono"))
    
    fig.update_layout(
        title=dict(
            text="PORTFOLIO EQUITY CURVE",
            font=dict(size=14, color="#ffffff", family="Outfit"),
            x=0, y=0.95
        ),
        xaxis=dict(
            gridcolor="#1f293d", 
            linecolor="#1f293d", 
            tickfont=dict(color="#8892b0", family="JetBrains Mono", size=10),
            title=dict(text="Timeline", font=dict(color="#8892b0", family="Outfit", size=11))
        ),
        yaxis=dict(
            gridcolor="#1f293d", 
            linecolor="#1f293d", 
            tickfont=dict(color="#8892b0", family="JetBrains Mono", size=10),
            title=dict(text="Value ($)", font=dict(color="#8892b0", family="Outfit", size=11))
        ),
        paper_bgcolor="#111625",
        plot_bgcolor="#111625",
        margin=dict(l=50, r=20, t=50, b=40),
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig


def _regime_bar(sim: Optional[pd.DataFrame]):
    import plotly.graph_objects as go
    names = list(REGIME_COLOURS.keys())
    probs = [0.25, 0.25, 0.25, 0.25]
    if sim is not None and "regime" in sim.columns:
        counts = sim["regime"].value_counts(normalize=True)
        probs  = [float(counts.get(n, 0)) for n in names]
    
    clean_names = [n.replace("_"," ").upper() for n in names]
    fig = go.Figure(go.Bar(
        x=clean_names,
        y=probs,
        marker=dict(
            color=list(REGIME_COLOURS.values()),
            line=dict(color="#111625", width=1.5)
        ),
        text=[f"{p:.1%}" for p in probs], textposition="outside",
        textfont=dict(color="#ffffff", family="JetBrains Mono", size=11)
    ))
    latest_regime = sim["regime"].iloc[-1] if (sim is not None and len(sim) > 0 and "regime" in sim.columns) else "?"
    latest_regime_str = latest_regime.replace('_',' ').upper()
    
    fig.update_layout(
        title=dict(
            text=f"REGIME FREQUENCY (CURRENT: {latest_regime_str})",
            font=dict(size=14, color="#ffffff", family="Outfit"),
            x=0, y=0.95
        ),
        xaxis=dict(
            linecolor="#1f293d", 
            tickfont=dict(color="#8892b0", family="Outfit", size=11)
        ),
        yaxis=dict(
            gridcolor="#1f293d", 
            linecolor="#1f293d", 
            tickfont=dict(color="#8892b0", family="JetBrains Mono", size=10),
            range=[0, max(probs) * 1.25 if max(probs) > 0 else 1.1]
        ),
        paper_bgcolor="#111625",
        plot_bgcolor="#111625",
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_dark"
    )
    return fig


def _positions_fig(state: dict):
    import plotly.graph_objects as go
    positions = state.get("positions", {})
    if not positions or all(q == 0 for q in positions.values()):
        fig = go.Figure()
        fig.add_annotation(
            text="NO ACTIVE OPEN POSITIONS", 
            x=0.5, y=0.5,
            xref="paper", yref="paper", 
            showarrow=False,
            font=dict(color="#8892b0", size=12, family="JetBrains Mono")
        )
    else:
        syms = list(positions.keys())
        qtys = [positions[s] for s in syms]
        fig = go.Figure(go.Bar(
            x=syms, y=qtys,
            marker_color=["#00e676" if q>0 else "#ff3860" for q in qtys],
            text=[f"{q:+.4f}" for q in qtys], textposition="outside",
            textfont=dict(color="#ffffff", family="JetBrains Mono", size=10)
        ))
    fig.update_layout(
        title=dict(
            text="OPEN POSITIONS EXPOSURE",
            font=dict(size=14, color="#ffffff", family="Outfit"),
            x=0, y=0.95
        ),
        xaxis=dict(
            linecolor="#1f293d", 
            tickfont=dict(color="#8892b0", family="Outfit", size=11)
        ),
        yaxis=dict(
            gridcolor="#1f293d", 
            linecolor="#1f293d", 
            tickfont=dict(color="#8892b0", family="JetBrains Mono", size=10),
            title=dict(text="Units", font=dict(color="#8892b0", family="Outfit", size=11))
        ),
        paper_bgcolor="#111625",
        plot_bgcolor="#111625",
        margin=dict(l=50, r=20, t=50, b=40),
        template="plotly_dark"
    )
    return fig


def _regime_pie(sim: Optional[pd.DataFrame]):
    import plotly.graph_objects as go
    if sim is None or "regime" not in sim.columns or sim.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="AWAITING SIMULATION DATA", 
            x=0.5, y=0.5,
            xref="paper", yref="paper", 
            showarrow=False,
            font=dict(color="#8892b0", size=12, family="JetBrains Mono")
        )
        fig.update_layout(
            title=dict(
                text="REGIME DISTRIBUTION",
                font=dict(size=14, color="#ffffff", family="Outfit"),
                x=0, y=0.95
            ),
            paper_bgcolor="#111625",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        return fig
    counts = sim["regime"].value_counts()
    fig = go.Figure(go.Pie(
        labels=[r.replace("_"," ").upper() for r in counts.index],
        values=counts.values,
        marker=dict(
            colors=[REGIME_COLOURS.get(r,"#8892b0") for r in counts.index],
            line=dict(color="#111625", width=2)
        ),
        hole=0.45,
        textinfo="percent+label",
        textfont=dict(family="Outfit", size=11)
    ))
    fig.update_layout(
        title=dict(
            text="REGIME DISTRIBUTION (HISTORIC)",
            font=dict(size=14, color="#ffffff", family="Outfit"),
            x=0, y=0.95
        ),
        paper_bgcolor="#111625",
        margin=dict(l=20, r=20, t=50, b=20),
        template="plotly_dark",
        showlegend=False
    )
    return fig


def _trade_table(trades: Optional[pd.DataFrame]):
    from dash import html
    if trades is None or trades.empty:
        return html.Div(
            "NO TRADE ACTIVITY DETECTED", 
            style={
                "color":"#8892b0", "fontFamily":"'JetBrains Mono', monospace", 
                "fontSize":"13px", "textAlign":"center", "padding":"40px",
                "background":"#111625", "border":"1px solid #1f293d", "borderRadius":"6px"
            }
        )
    recent = trades.tail(10).iloc[::-1]
    th = lambda t: html.Th(t, style={
        "padding":"12px 16px", "textAlign":"left", "fontSize":"11px", 
        "color":"#8892b0", "borderBottom":"1px solid #1f293d", 
        "fontFamily":"'Outfit', sans-serif", "textTransform":"uppercase", "letterSpacing":"1px"
    })
    header = html.Tr([th(c) for c in ["Time","Symbol","Direction","Qty","Price","Value","Portfolio"]])
    rows = []
    for _, row in recent.iterrows():
        d = str(row.get("direction","")).upper()
        col = "#00e676" if "LONG" in d or "BUY" in d else ("#ff3860" if "SHORT" in d or "SELL" in d else "#8892b0")
        
        td = lambda v, **s: html.Td(v, style={
            "padding":"12px 16px", "fontSize":"12px", "color":"#ffffff",
            "fontFamily":"'JetBrains Mono', monospace", "borderBottom":"1px solid #1f293d", **s
        })
        
        badge_bg = "rgba(0, 230, 118, 0.1)" if "LONG" in d or "BUY" in d else "rgba(255, 56, 96, 0.1)"
        badge = html.Span(d, style={
            "color": col, "background": badge_bg, "padding": "2px 6px", 
            "borderRadius": "4px", "fontWeight": "bold", "fontSize": "10px"
        })
        
        rows.append(html.Tr([
            td(str(row.get("timestamp",""))[:16]),
            td(str(row.get("symbol","")), fontWeight="600"),
            td(badge),
            td(f"{row.get('quantity',0):+.4f}"),
            td(f"${row.get('price',0):,.4f}"),
            td(f"${row.get('trade_value',0):,.2f}"),
            td(f"${row.get('portfolio_val',0):,.2f}", fontWeight="600", color="#3b82f6"),
        ], className="trade-row", style={"background": "#111625"}))
        
    return html.Table([header]+rows, style={
        "width":"100%","borderCollapse":"collapse",
        "background":"#111625","borderRadius":"6px",
        "border":"1px solid #1f293d", "overflow":"hidden"
    })