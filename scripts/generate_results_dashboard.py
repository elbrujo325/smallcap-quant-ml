#!/usr/bin/env python3
"""Generate a small selection of backtest charts for README showcase."""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import DATA, FEATURES, RISK, BACKTEST
from src.data import load_ohlc_from_yfinance
from src.features import add_all_features
from src.risk import run_backtest_loop, calculate_performance_metrics, find_optimal_coef_sl

OUTPUT_DIR = REPO_ROOT / "docs" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_strategies(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        raw_strategies = json.load(fh)

    strategies = []
    for item in raw_strategies:
        conditions = item.get("conditions", [])
        if isinstance(conditions, list):
            parsed = []
            for cond in conditions:
                if isinstance(cond, str):
                    m = re.match(r"^(\w+)\s*(<=|>=|<|>|==)\s*([\d.+-]+)$", cond)
                    if m:
                        feature, operator, threshold = m.groups()
                        parsed.append((feature, float(threshold), operator))
                    else:
                        parsed.append((cond, None, None))
                else:
                    parsed.append(cond)
            strategies.append({
                "id": item.get("rule_id"),
                "name": f"Rule {item.get('rule_id')}",
                "conditions": parsed,
            })
        else:
            strategies.append(item)
    return strategies


def build_results_df() -> pd.DataFrame:
    strategies = load_strategies(REPO_ROOT / "data" / "strategies_candidates.json")
    universe_path = REPO_ROOT / "config" / "tickers_smallcap.txt"
    with open(universe_path, "r", encoding="utf-8") as fh:
        tickers = [
            line.strip()
            for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]

    rows = []
    for ticker in tickers:
        try:
            df = load_ohlc_from_yfinance(ticker=ticker, period=DATA.period, interval=DATA.interval)
            if df.empty:
                continue

            df = add_all_features(df)
            if "ATR" not in df.columns:
                df["ATR"] = df["ATR_50"]
            df = df.dropna(subset=["ATR", "VWAP_20", "EMA_20", "EMA_50", "SMA_10"]).reset_index(drop=True)
            if df.empty:
                continue

            csl = find_optimal_coef_sl(
                df=df,
                n_samples=RISK.n_samples_csl,
                lookforward_window=RISK.lookforward_window,
                atr_period=FEATURES.atr_period,
                price_min=DATA.price_min,
                price_max=DATA.price_max,
                momentum_k=FEATURES.er_k,
                momentum_threshold=FEATURES.er_threshold,
                admission_range=RISK.csl_admission_range,
                seed=RISK.seed_csl,
            )
            if csl is None:
                continue

            for strategy in strategies:
                conditions = strategy.get("conditions", [])
                if not conditions:
                    continue

                signal = pd.Series(True, index=df.index)
                for feature, threshold, operator in conditions:
                    if not isinstance(feature, str) or not isinstance(operator, str) or threshold is None:
                        continue
                    feature_values = df[feature]
                    if operator == "<":
                        signal = signal & (feature_values < threshold)
                    elif operator == "<=":
                        signal = signal & (feature_values <= threshold)
                    elif operator == ">":
                        signal = signal & (feature_values > threshold)
                    elif operator == ">=":
                        signal = signal & (feature_values >= threshold)
                    elif operator == "==":
                        signal = signal & (feature_values == threshold)

                if signal.sum() == 0:
                    continue

                trades = run_backtest_loop(
                    df=df,
                    entry_signal=signal,
                    c_sl=csl,
                    c_tp=None,
                    max_bars=BACKTEST.max_bars,
                    risk_per_trade=RISK.risk_per_trade,
                    capital=BACKTEST.capital,
                )
                metrics = calculate_performance_metrics(trades, BACKTEST.capital)
                pnls = trades["PnL"].tolist() if "PnL" in trades.columns else []
                cumulative = np.cumsum(pnls) if pnls else []
                corr = float(np.corrcoef(np.arange(1, len(cumulative) + 1), cumulative)[0, 1]) if len(cumulative) > 1 else 0.0

                rows.append({
                    "ticker": ticker,
                    "strategy_id": strategy.get("id"),
                    "strategy_name": strategy.get("name"),
                    "conditions": conditions,
                    "csl": csl,
                    "n_trades": len(trades),
                    "trade_pnls": pnls,
                    "cumulative_pnls": cumulative,
                    "total_pnl": metrics.get("PnL Total", 0),
                    "return_pct": metrics.get("Retorno Total (%)", 0),
                    "win_rate": metrics.get("Win Rate", 0),
                    "profit_factor": metrics.get("Profit Factor", 0),
                    "max_drawdown_pct": metrics.get("Max Drawdown (%)", 0),
                    "sharpe": metrics.get("Sharpe Ratio", 0),
                    "corr_trade_vs_pnl": corr,
                })
        except Exception as exc:
            print(f"No se pudo procesar {ticker}: {exc}")

    return pd.DataFrame(rows)


def plot_series(ax, row, label_prefix: str = ""):
    cumulative_value = row.get("cumulative_pnls") if isinstance(row, dict) else None
    if cumulative_value is None:
        return
    if isinstance(cumulative_value, np.ndarray):
        cumulative = cumulative_value.tolist()
    else:
        cumulative = list(cumulative_value)
    if len(cumulative) < 2:
        return
    x = range(1, len(cumulative) + 1)
    ax.plot(x, cumulative, color="#2ca02c", linewidth=2.2, alpha=0.9)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL acumulado")
    ax.set_title(
        f"{label_prefix}{row['ticker']} | Rule {row['strategy_id']}\n"
        f"PnL={row['total_pnl']:.0f} | Corr={row['corr_trade_vs_pnl']:.2f}"
    )
    ax.grid(alpha=0.25)


def save_selected_charts(results_df: pd.DataFrame) -> list:
    filtered = results_df[
        (results_df["total_pnl"] > 0)
        & (results_df["corr_trade_vs_pnl"] > 0.3)
        & (results_df["max_drawdown_pct"] > -35)
    ].copy()

    if filtered.empty:
        filtered = results_df[(results_df["total_pnl"] > 0) & (results_df["corr_trade_vs_pnl"] > 0.1)].copy()

    filtered = filtered.sort_values(["total_pnl", "corr_trade_vs_pnl"], ascending=False)
    top_rows = filtered.head(4).copy()

    for idx, row in enumerate(top_rows.itertuples(index=False), start=1):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_series(ax, row._asdict(), label_prefix="")
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"result_{idx:02d}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Combined figure for README hero image
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, row in zip(axes, top_rows.itertuples(index=False)):
        plot_series(ax, row._asdict(), label_prefix="")

    for ax in axes[len(top_rows):]:
        ax.axis("off")

    fig.tight_layout()
    combined_path = OUTPUT_DIR / "results_dashboard.png"
    fig.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return [OUTPUT_DIR / f"result_{i:02d}.png" for i in range(1, min(4, len(top_rows)) + 1)] + [combined_path]


def main():
    results_df = build_results_df()
    if results_df.empty:
        raise RuntimeError("No se pudieron generar resultados de backtest")

    results_df.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    image_paths = save_selected_charts(results_df)
    print(f"Resultados generados: {len(results_df)} combinaciones")
    print("Rutas de imagen:")
    for path in image_paths:
        print(path)

    print("\nTop 10 seleccionados:")
    filtered = results_df[
        (results_df["total_pnl"] > 0)
        & (results_df["corr_trade_vs_pnl"] > 0.3)
        & (results_df["max_drawdown_pct"] > -35)
    ].copy()
    if filtered.empty:
        filtered = results_df[(results_df["total_pnl"] > 0) & (results_df["corr_trade_vs_pnl"] > 0.1)].copy()
    print(filtered[["ticker", "strategy_id", "total_pnl", "return_pct", "corr_trade_vs_pnl", "max_drawdown_pct"]].sort_values(["total_pnl", "corr_trade_vs_pnl"], ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
