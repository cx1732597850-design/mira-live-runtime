import json
from pathlib import Path
import pandas as pd
import trend_runtime_v2 as rt

def main():
    df=rt.download([2023,2024,2025,2026])
    ds=rt.build_states(df)
    results={}
    for mode in rt.MODES:
        d=ds[(ds.season==2026)&(ds['mode']==mode)&ds[rt.BASELINE_COLS].notna().all(axis=1)].copy()
        if d.empty: raise RuntimeError(f'no 2026 baseline-ready row for {mode}')
        row=d.iloc[-1]
        models,feats,_=rt.fit_models(ds[ds.season.isin([2023,2024,2025])],mode,[2023,2024,2025])
        payload={'mode':mode}
        for c in feats:
            v=row[c]
            if pd.notna(v): payload[c]=float(v)
        for k in ['cur_margin','cur_total','poss_total','poss_diff','elapsed_sec','pace_poss_per_min','a_score','a_poss','b_score','b_poss','pregame_net_diff','pregame_pace_diff','pregame_off_diff','pregame_def_diff','pregame_games_min']:
            if k not in payload and k in row.index and pd.notna(row[k]): payload[k]=float(row[k])
        out=rt.infer(payload)
        expected='PREVIEW_CHECK' if mode=='Q1_END' else 'TREND_FINAL'
        if out.get('status')!=expected: raise RuntimeError(f'{mode} infer state failed: expected {expected}, got {out}')
        for k in ['pred_margin','pred_total','direction','trend_strength','betting_layer']:
            if k not in out: raise RuntimeError(f'{mode} missing {k}')
        if out['betting_layer']!='DISABLED': raise RuntimeError('betting layer leak')
        results[mode]={'game_id':str(row.game_id),'team_a':str(row.team_a),'team_b':str(row.team_b),'output':out}
    bad={'mode':'Q1_2P5','cur_margin':0,'cur_total':0,'poss_total':1,'poss_diff':0,'elapsed_sec':150,'pace_poss_per_min':0.4,'a_score':0,'a_poss':1,'b_score':0,'b_poss':0,'pregame_net_diff':0,'pregame_pace_diff':0,'pregame_off_diff':0,'pregame_def_diff':0,'pregame_games_min':2}
    neg=rt.infer(bad)
    if neg.get('status')!='NOT_READY' or neg.get('reason')!='INSUFFICIENT_2026_PREGAME_BASELINE': raise RuntimeError(f'negative baseline canary failed: {neg}')
    results['negative_baseline_canary']=neg
    Path('artifacts/runtime_v2_canary').mkdir(parents=True,exist_ok=True)
    json.dump({'status':'PASS','results':results},open('artifacts/runtime_v2_canary/canary.json','w'),indent=2)
    print(json.dumps({'status':'PASS','results':results},indent=2))

if __name__=='__main__': main()
