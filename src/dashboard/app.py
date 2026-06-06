"""Streamlit analytical dashboard for the backtesting platform.

Run with::

    streamlit run src/dashboard/app.py

The dashboard connects to the FastAPI backend or directly imports the
Python modules for standalone operation. Visualisations are built with
Plotly for interactivity.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QB Risk Infra — Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.title("Configuration")

data_dir = st.sidebar.text_input(
    "CSV Data Directory", value="data/raw", help="Directory containing per-symbol CSV files"
)

strategy = st.sidebar.selectbox(
    "Strategy",
    ["moving_average_crossover", "mean_reversion", "momentum", "volatility_breakout"],
)

symbol_input = st.sidebar.text_input("Symbol", value="AAPL")

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", value=date(2019, 1, 1))
end_date = col2.date_input("End", value=date(2023, 12, 31))

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")

if strategy == "moving_average_crossover":
    fast = st.sidebar.slider("Fast EMA", 5, 50, 20)
    slow = st.sidebar.slider("Slow EMA", 20, 200, 50)
    params = {"fast_period": fast, "slow_period": slow}
elif strategy == "mean_reversion":
    lookback = st.sidebar.slider("Lookback", 10, 100, 30)
    entry_z = st.sidebar.slider("Entry Z", 1.0, 4.0, 2.0, step=0.1)
    exit_z = st.sidebar.slider("Exit Z", 0.1, 2.0, 0.5, step=0.1)
    params = {"lookback": lookback, "entry_z": entry_z, "exit_z": exit_z}
elif strategy == "momentum":
    lookback = st.sidebar.slider("Lookback", 20, 252, 63)
    vol_scaled = st.sidebar.checkbox("Volatility Scaled", value=True)
    params = {"lookback": lookback, "volatility_scaled": vol_scaled}
elif strategy == "volatility_breakout":
    bb_period = st.sidebar.slider("BB Period", 10, 50, 20)
    bb_std = st.sidebar.slider("BB Std", 1.0, 4.0, 2.0, step=0.1)
    params = {"bb_period": bb_period, "bb_std": bb_std}
else:
    params = {}

st.sidebar.markdown("---")
initial_capital = st.sidebar.number_input("Initial Capital", value=100_000, step=10_000)
commission_bps = st.sidebar.slider("Commission (bps)", 0, 20, 5)
slippage_bps = st.sidebar.slider("Slippage (bps)", 0, 10, 2)

run_btn = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("Quantitative Backtesting & Risk Analytics")
st.caption(
    "Event-driven backtesting engine with walk-forward validation, "
    "Monte Carlo stress testing, and portfolio risk analytics."
)

if not run_btn:
    st.info(
        "Configure strategy parameters in the sidebar and click **Run Backtest** to start."
    )
    st.stop()

# ── Run backtest ──────────────────────────────────────────────────────────────
with st.spinner("Running backtest..."):
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

        from src.core.config import settings
        from src.ingestion.csv_adapter import CSVIngestionAdapter
        from src.strategies.registry import get_strategy
        from src.backtesting.engine import BacktestEngine
        from src.backtesting.order_router import OrderRouter
        from src.backtesting.portfolio import Portfolio
        from src.backtesting.metrics import compute_all_metrics, drawdown_series, rolling_volatility
        from src.risk.analytics import rolling_var, rolling_cvar

        adapter = CSVIngestionAdapter(data_dir=data_dir)
        data = adapter.fetch(
            symbol=symbol_input.upper(),
            start=datetime.combine(start_date, datetime.min.time()),
            end=datetime.combine(end_date, datetime.min.time()),
        )

        strategy_obj = get_strategy(strategy, params)
        router_obj = OrderRouter(commission_bps=commission_bps, slippage_bps=slippage_bps)
        portfolio = Portfolio(initial_capital=float(initial_capital))
        engine = BacktestEngine(
            strategy=strategy_obj,
            order_router=router_obj,
            portfolio=portfolio,
        )
        result = engine.run(data=data, symbol=symbol_input.upper())

    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        st.exception(exc)
        st.stop()

# ── Metrics row ───────────────────────────────────────────────────────────────
m = result.metrics
st.subheader("Performance Summary")
cols = st.columns(7)
metric_pairs = [
    ("Ann. Return", f"{m.get('annualised_return', 0):.2%}"),
    ("Ann. Volatility", f"{m.get('annualised_volatility', 0):.2%}"),
    ("Sharpe", f"{m.get('sharpe_ratio', 0):.3f}"),
    ("Sortino", f"{m.get('sortino_ratio', 0):.3f}"),
    ("Max Drawdown", f"{m.get('max_drawdown', 0):.2%}"),
    ("VaR 95%", f"{m.get('var_95', 0):.2%}"),
    ("CVaR 95%", f"{m.get('cvar_95', 0):.2%}"),
]
for col, (label, value) in zip(cols, metric_pairs):
    col.metric(label, value)

st.markdown("---")

# ── Equity curve & drawdown ───────────────────────────────────────────────────
equity = result.equity_curve
returns = result.returns
dd = drawdown_series(equity)

fig_equity = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
    subplot_titles=["Portfolio Equity Curve", "Drawdown"]
)

fig_equity.add_trace(
    go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Equity",
        line=dict(color="#1f77b4", width=1.5),
    ),
    row=1, col=1,
)

fig_equity.add_trace(
    go.Scatter(
        x=dd.index, y=dd.values * 100,
        mode="lines", name="Drawdown (%)",
        fill="tozeroy", line=dict(color="#d62728", width=1),
        fillcolor="rgba(214, 39, 40, 0.15)",
    ),
    row=2, col=1,
)

fig_equity.update_layout(
    height=500, showlegend=True,
    legend=dict(orientation="h", y=1.02),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig_equity.update_yaxes(title_text="Equity ($)", row=1, col=1)
fig_equity.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

st.plotly_chart(fig_equity, use_container_width=True)

# ── Rolling volatility & risk ─────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Rolling Volatility (21d)")
    from src.backtesting.metrics import rolling_volatility as _roll_vol
    if callable(_roll_vol):
        roll_vol = returns.rolling(21).std(ddof=1) * np.sqrt(252) * 100
    else:
        roll_vol = returns.rolling(21).std() * np.sqrt(252) * 100

    fig_vol = go.Figure(go.Scatter(
        x=roll_vol.index, y=roll_vol.values,
        mode="lines", name="Realised Vol (%)",
        line=dict(color="#ff7f0e", width=1.5),
        fill="tozeroy", fillcolor="rgba(255, 127, 14, 0.1)",
    ))
    fig_vol.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Annualised Vol (%)",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with col_b:
    st.subheader("Rolling VaR 95% (63d)")
    rv = returns.rolling(63, min_periods=30).quantile(0.05) * -100

    fig_var = go.Figure(go.Scatter(
        x=rv.index, y=rv.values,
        mode="lines", name="VaR 95% (%)",
        line=dict(color="#9467bd", width=1.5),
        fill="tozeroy", fillcolor="rgba(148, 103, 189, 0.1)",
    ))
    fig_var.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Daily VaR (%)",
    )
    st.plotly_chart(fig_var, use_container_width=True)

# ── Return distribution ───────────────────────────────────────────────────────
st.subheader("Return Distribution")
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=returns.dropna().values * 100,
    nbinsx=80,
    name="Daily Returns",
    marker_color="#2ca02c",
    opacity=0.75,
))
fig_hist.add_vline(
    x=float(-m.get("var_95", 0)) * 100,
    line_dash="dash", line_color="red",
    annotation_text="-VaR 95%", annotation_position="top right",
)
fig_hist.update_layout(
    height=300, margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title="Daily Return (%)",
    yaxis_title="Frequency",
    bargap=0.05,
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Monte Carlo stress test ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Monte Carlo Simulation (1,000 paths, 252-day horizon)")

with st.spinner("Running Monte Carlo simulation..."):
    try:
        from src.montecarlo.simulator import MonteCarloSimulator

        mc_sim = MonteCarloSimulator(
            n_simulations=500,
            horizon_days=252,
            method="bootstrap",
            confidence_levels=[0.95, 0.99],
            random_seed=42,
        )
        mc_result = mc_sim.simulate(returns, initial_value=float(equity.iloc[-1]))

        # Fan chart of paths
        horizon = np.arange(1, mc_result.horizon_days + 1)
        p5 = np.percentile(mc_result.paths, 5, axis=0)
        p25 = np.percentile(mc_result.paths, 25, axis=0)
        p50 = np.percentile(mc_result.paths, 50, axis=0)
        p75 = np.percentile(mc_result.paths, 75, axis=0)
        p95 = np.percentile(mc_result.paths, 95, axis=0)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([horizon, horizon[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill="toself", fillcolor="rgba(31, 119, 180, 0.1)",
            line=dict(color="rgba(255,255,255,0)"), name="5th–95th pct",
        ))
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([horizon, horizon[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself", fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(color="rgba(255,255,255,0)"), name="25th–75th pct",
        ))
        fig_mc.add_trace(go.Scatter(
            x=horizon, y=p50, mode="lines",
            line=dict(color="#1f77b4", width=2), name="Median path",
        ))
        fig_mc.add_hline(
            y=float(equity.iloc[-1]),
            line_dash="dot", line_color="gray",
            annotation_text="Current equity",
        )
        fig_mc.update_layout(
            height=380, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Trading Days Forward",
            yaxis_title="Portfolio Value ($)",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

        mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
        mc_col1.metric("VaR 95% (1yr)", f"{mc_result.var.get(0.95, 0):.2%}")
        mc_col2.metric("CVaR 95% (1yr)", f"{mc_result.cvar.get(0.95, 0):.2%}")
        mc_col3.metric("Median Terminal", f"${mc_result.percentiles[50]:,.0f}")
        mc_col4.metric("5th pct Terminal", f"${mc_result.percentiles[5]:,.0f}")

    except Exception as exc:
        st.warning(f"Monte Carlo simulation skipped: {exc}")

# ── Fill log ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Trade Log ({len(result.fills)} fills)")

if result.fills:
    fill_data = [
        {
            "Timestamp": str(f.timestamp.date()),
            "Side": f.order.side.value,
            "Quantity": round(f.fill_quantity, 4),
            "Fill Price": round(f.fill_price, 4),
            "Commission": round(f.commission, 2),
            "Slippage": round(f.slippage, 2),
        }
        for f in result.fills[:200]  # cap display at 200 rows
    ]
    st.dataframe(pd.DataFrame(fill_data), use_container_width=True)
else:
    st.info("No fills generated in this backtest run.")

# ── Full metrics table ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Full Metrics")

metrics_display = {
    "Annualised Return": f"{m.get('annualised_return', 0):.4%}",
    "Annualised Volatility": f"{m.get('annualised_volatility', 0):.4%}",
    "Sharpe Ratio": f"{m.get('sharpe_ratio', 0):.4f}",
    "Sortino Ratio": f"{m.get('sortino_ratio', 0):.4f}",
    "Calmar Ratio": f"{m.get('calmar_ratio', 0):.4f}",
    "Max Drawdown": f"{m.get('max_drawdown', 0):.4%}",
    "Hit Rate": f"{m.get('hit_rate', 0):.4%}",
    "Profit Factor": f"{m.get('profit_factor', 0):.4f}",
    "Skewness": f"{m.get('skewness', 0):.4f}",
    "Kurtosis": f"{m.get('kurtosis', 0):.4f}",
    "VaR 95%": f"{m.get('var_95', 0):.4%}",
    "CVaR 95%": f"{m.get('cvar_95', 0):.4%}",
    "VaR 99%": f"{m.get('var_99', 0):.4%}",
    "CVaR 99%": f"{m.get('cvar_99', 0):.4%}",
    "N Periods": str(int(m.get('n_periods', 0))),
}

st.dataframe(
    pd.DataFrame.from_dict(metrics_display, orient="index", columns=["Value"]),
    use_container_width=True,
)
