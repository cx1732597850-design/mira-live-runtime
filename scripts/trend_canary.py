import json
from pathlib import Path
import numpy as np
import pandas as pd

IN = Path('artifacts/fit_oos/oos_predictions_2025.csv')
OUT = Path('artifacts/trend_verification')
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    'no_clear_trend_lt': 3.0,
    'strong_trend_ge': 5.0,
}

MODES = ['Q1_2P5','Q1_END','Q2_2P5']


def classify(abs_margin: float) -> str:
    if abs_margin < THRESHOLDS['no_clear_trend_lt']:
        return 'NO_CLEAR_TREND'
    if abs_margin < THRESHOLDS['strong_trend_ge']:
        return 'WEAK_MODERATE_TREND'
    return 'STRONG_TREND'


def accuracy(df):
    nz = df[df.target_margin != 0].copy()
    if len(nz) == 0:
        return None
    return float((np.sign(nz.pred_margin) == np.sign(nz.target_margin)).mean())


def main():
    if not IN.exists():
        raise RuntimeError('missing OOS predictions; run fit_validate.py first')
    df = pd.read_csv(IN)
    required = {'season','game_id','mode','target_margin','pred_margin','target_total','pred_total'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f'missing required columns: {missing}')
    if set(df['season'].unique()) != {2025}:
        raise RuntimeError('canary requires untouched 2025 OOS only')
    if sorted(df['mode'].unique()) != sorted(MODES):
        raise RuntimeError('mode set mismatch')

    df['abs_pred_margin'] = df['pred_margin'].abs()
    df['trend_state'] = df['abs_pred_margin'].map(classify)

    # Boundary canaries: exact contract behavior.
    boundary = {
        '2.9999': classify(2.9999),
        '3.0': classify(3.0),
        '4.9999': classify(4.9999),
        '5.0': classify(5.0),
    }
    expected = {
        '2.9999':'NO_CLEAR_TREND',
        '3.0':'WEAK_MODERATE_TREND',
        '4.9999':'WEAK_MODERATE_TREND',
        '5.0':'STRONG_TREND',
    }
    if boundary != expected:
        raise RuntimeError(f'boundary canary failed: {boundary}')

    report = {
        'pipeline':'MIRA-WNBA-LIVE-V1.0-TREND-STRENGTH-VERIFICATION',
        'oos_season':2025,
        'thresholds':THRESHOLDS,
        'boundary_canary':boundary,
        'betting_layer_enabled':False,
        'market_archive_required':False,
        'modes':{},
        'fail_closed_contract':{
            'critical_state_missing':'NOT_READY',
            'stale_or_inconsistent_clock_score':'NOT_READY',
            'market_quote_missing':'NOT_A_BLOCKER_FOR_TREND_ONLY'
        }
    }

    for mode in MODES:
        m = df[df['mode']==mode].copy()
        mode_rep = {'n':int(len(m)), 'overall_direction_accuracy_ex_ties':accuracy(m), 'states':{}}
        for state in ['NO_CLEAR_TREND','WEAK_MODERATE_TREND','STRONG_TREND']:
            s = m[m['trend_state']==state]
            mode_rep['states'][state] = {
                'n':int(len(s)),
                'direction_accuracy_ex_ties':accuracy(s),
                'margin_mae':float(np.mean(np.abs(s.target_margin-s.pred_margin))) if len(s) else None,
                'total_mae':float(np.mean(np.abs(s.target_total-s.pred_total))) if len(s) else None,
            }
        report['modes'][mode] = mode_rep

    # Structural canaries: every row must classify exactly once, and no betting fields are created.
    if df['trend_state'].isna().any():
        raise RuntimeError('unclassified trend row')
    banned = {'market_line','price','edge','stake','decision','max_playable_price','pnl'}
    if banned.intersection(df.columns):
        raise RuntimeError('betting-layer field leaked into trend-only output')

    report['status'] = 'TREND_STRENGTH_GATE_CANARIES_PASS'
    df.to_csv(OUT/'trend_classified_oos_2025.csv', index=False)
    with open(OUT/'trend_verification_report.json','w') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
