#!/usr/bin/env python3
"""
batch_calibrate.py - Calibración de riesgo batch para universos de tickers.

Carga lista de tickers, calibra Csl y verifica BP para cada uno.
Output: CSV con activos admitidos (Csl + BP) y todos los resultados.

Uso:
    python scripts/batch_calibrate.py
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configurar rutas
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import DATA, FEATURES, RISK, get_universe_tickers
from src.data import load_ohlc_from_yfinance, validate_ohlc_data
from src.risk import find_optimal_coef_sl, calculate_buying_power_distribution


def main():
    print(f"[*] Iniciando calibración batch: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Universo: {len(get_universe_tickers())} tickers definidos")
    
    tickets = get_universe_tickers()
    results = []
    admitted = []
    
    for i, ticker in enumerate(tickets):
        print(f"\n[{i+1}/{len(tickets)}] Procesando {ticker}...", end=" ")
        sys.stdout.flush()
        
        try:
            # 1. Cargar datos
            df = load_ohlc_from_yfinance(ticker, period=DATA.period, interval=DATA.interval)
            
            # 2. Validar
            if not validate_ohlc_data(df, min_rows=500):  # mínimo para tener 500 barras de ATR(50)
                print(f"❌ Datos insuficientes ({len(df)} barras)")
                results.append({
                    'ticker': ticker, 'error': 'insufficient_data', 'csl': None, 'bp_median': None, 'bp_max': None, 'admitted': False
                })
                continue
            
            # 3. Calibrar Csl (Paso 1)
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
                seed=RISK.seed_csl
            )
            
            if csl is None:
                print(f"❌ Csl no admitido")
                results.append({
                    'ticker': ticker, 'csl': None, 'bp_median': None, 'bp_max': None, 'admitted': False, 'error': 'csl_rejected'
                })
                continue
            
            # 4. Verificar BP (Paso 2, con Csl fijo)
            bp_res = calculate_buying_power_distribution(
                df=df,
                c_sl=csl,
                n_samples=RISK.n_samples_bp,
                lookforward_window=RISK.lookforward_window,
                atr_period=FEATURES.atr_period,
                price_min=DATA.price_min,
                price_max=DATA.price_max,
                risk_per_trade=RISK.risk_per_trade,
                seed=RISK.seed_bp
            )
            
            if not bp_res['admitted']:
                print(f"❌ BP no admitido (med={bp_res['median_bp']:.0f}, max={bp_res['max_bp']:.0f})")
                results.append({
                    'ticker': ticker, 'csl': csl, 'bp_median': bp_res['median_bp'], 
                    'bp_max': bp_res['max_bp'], 'admitted': False, 'error': 'bp_rejected'
                })
                continue
            
            # ✅ ADIMITIDO
            print(f"✅ Aceptado (Csl={csl:.3f}, BP_med=${bp_res['median_bp']:.0f})")
            admitted.append({
                'ticker': ticker, 'csl': csl, 'bp_median': bp_res['median_bp'], 
                'bp_max': bp_res['max_bp'], 'n_samples': bp_res['n_valid_samples']
            })
            results.append({
                'ticker': ticker, 'csl': csl, 'bp_median': bp_res['median_bp'], 
                'bp_max': bp_res['max_bp'], 'admitted': True
            })
            
        except Exception as e:
            print(f"⚠️ Error: {str(e)[:50]}")
            results.append({'ticker': ticker, 'error': str(e), 'admitted': False})
    
    # Guardar resultados
    out_dir = REPO_ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "calibration_all.csv", index=False)
    
    if admitted:
        admitted_df = pd.DataFrame(admitted)
        admitted_df = admitted_df.sort_values('csl')  # ordenar por Csl
        admitted_df.to_csv(out_dir / "universe_admitted.csv", index=False)
        print(f"\n[+] TOTAL: {len(admitted)}/{len(tickets)} admitidos")
        print(f"[+] Resultados guardados en: {out_dir}")
    else:
        print(f"\n[!] 0 tickers admitidos. Revisa los parámetros o el universo.")
        results_df.to_csv(out_dir / "calibration_all.csv", index=False)


if __name__ == "__main__":
    main()