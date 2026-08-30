import json
from pathlib import Path
import pandas as pd
import trend_runtime_v2 as v2
import trend_runtime_v3 as v3

CLOCK={'Q1_2P5':450.0,'Q1_END':0.0,'Q2_2P5':450.0}
PERIOD={'Q1_2P5':1,'Q1_END':1,'Q2_2P5':2}

def payload_from_row(row,mode):
    n=int(float(row['pregame_games_min']))
    p={
      'game_id':str(row.game_id),'source_event_id':str(row.game_id),'team_a_id':str(row.team_a),'team_b_id':str(row.team_b),
      'mode':mode,'period':PERIOD[mode],'clock_seconds_remaining':CLOCK[mode],
      'captured_at_et':'2026-08-29T19:32:00-04:00','execution_time_et':'2026-08-29T19:32:20-04:00','input_source':'CANARY_STRUCTURED_STATE',
      'baseline_provider':v3.BASELINE_PROVIDER,'baseline_asof_et':'2026-08-29T18:45:00-04:00','baseline_source_id':'CANARY_STATMUSE_GAME_LEVEL_ADVANCED','baseline_source_hash':v2.SOURCES[2026][1],
      'baseline_window_games_a':min(n,10),'baseline_window_games_b':min(n,10),'baseline_prior_games_a':n,'baseline_prior_games_b':n,
    }
    for k in v2.BASELINE_COLS:p[k]=float(row[k])
    p['pregame_net_diff']=p['pregame_off_diff']-p['pregame_def_diff']
    for pref in ['a','b']:
      for k in ['score','poss']+v2.STAT_COLS:p[f'{pref}_{k}']=float(row[f'{pref}_{k}'])
    return p

def require_not_ready(out,reason):
    if out.get('status')!='NOT_READY' or out.get('reason')!=reason: raise RuntimeError(f'expected {reason}, got {out}')

def main():
    df=v2.download([2023,2024,2025,2026]); ds=v2.build_states(df); results={}
    for mode in v2.MODES:
      d=ds[(ds.season==2026)&(ds['mode']==mode)&ds[v2.BASELINE_COLS].notna().all(axis=1)].copy()
      if d.empty: raise RuntimeError(f'no baseline-ready 2026 state: {mode}')
      row=d.iloc[-1]; p=payload_from_row(row,mode); out=v3.infer(p)
      expected_status='PREVIEW_CHECK' if mode=='Q1_END' else 'TREND_FINAL'
      if out.get('status')!=expected_status: raise RuntimeError(f'{mode} v3 failed {out}')
      if out.get('betting_layer')!='DISABLED' or out.get('input_contract')!='PASS' or out.get('baseline_contract')!='PASS': raise RuntimeError('contract leak')
      models,feats,_=v2.fit_models(ds[ds.season.isin([2023,2024,2025])],mode,[2023,2024,2025])
      xr={c:float(row[c]) for c in feats}; xr['pregame_net_diff']=p['pregame_net_diff']; x=pd.DataFrame([xr])[feats]
      pm=float(models['target_margin'].predict(x)[0]); pt=float(models['target_total'].predict(x)[0])
      if abs(pm-out['pred_margin'])>1e-8 or abs(pt-out['pred_total'])>1e-8: raise RuntimeError(f'FROZEN_BUNDLE_EQUIVALENCE_FAIL {mode} {pm} {out}')
      ps=dict(p)
      for pref in ['a','b']:
        ps[f'{pref}_fgm']=ps[f'{pref}_fg2m']+ps[f'{pref}_fg3m']; ps[f'{pref}_fga']=ps[f'{pref}_fg2a']+ps[f'{pref}_fg3a']
        del ps[f'{pref}_fg2a']; del ps[f'{pref}_fg2m']; del ps[f'{pref}_poss']
      sout=v3.infer(ps)
      if sout.get('status')!=expected_status or sout.get('possession_method_a')!='BOX_ESTIMATE_044': raise RuntimeError(f'SCREENSHOT_ADAPTER_FAIL {mode} {sout}')
      results[mode]={'game_id':str(row.game_id),'v3':out,'screenshot_adapter':{'status':sout['status'],'possession_method_a':sout['possession_method_a'],'possession_method_b':sout['possession_method_b']},'v2_equivalence_abs_error':{'margin':abs(pm-out['pred_margin']),'total':abs(pt-out['pred_total'])}}
    base=payload_from_row(ds[(ds.season==2026)&(ds['mode']=='Q1_2P5')&ds[v2.BASELINE_COLS].notna().all(axis=1)].iloc[-1],'Q1_2P5')
    x=dict(base); del x['a_fg3a']; require_not_ready(v3.infer(x),'MISSING_REQUIRED_FEATURES')
    x=dict(base); x['period']=2; require_not_ready(v3.infer(x),'WINDOW_MISMATCH')
    x=dict(base); x['clock_seconds_remaining']=300; require_not_ready(v3.infer(x),'WINDOW_MISMATCH')
    x=dict(base); x['execution_time_et']='2026-08-29T19:35:01-04:00'; require_not_ready(v3.infer(x),'STALE_LIVE_STATE')
    x=dict(base); x['team_b_id']=x['team_a_id']; require_not_ready(v3.infer(x),'TEAM_IDENTITY_MISMATCH')
    x=dict(base); x['a_score']=x['a_score']+1; require_not_ready(v3.infer(x),'STATE_IDENTITY_MISMATCH')
    x=dict(base); x['cur_margin']=999; require_not_ready(v3.infer(x),'STATE_IDENTITY_MISMATCH')
    x=dict(base); x['pregame_games_min']=2; require_not_ready(v3.infer(x),'BASELINE_PRIOR_GAMES_IDENTITY_MISMATCH')
    x=dict(base); x['baseline_window_games_a']=2; require_not_ready(v3.infer(x),'BASELINE_WINDOW_COUNT_INVALID')
    x=dict(base); x['baseline_prior_games_a']=2; x['baseline_window_games_a']=3; x['pregame_games_min']=2; require_not_ready(v3.infer(x),'BASELINE_PRIOR_COUNT_INVALID')
    x=dict(base); x['baseline_asof_et']='2026-08-27T00:00:00-04:00'; require_not_ready(v3.infer(x),'BASELINE_SOURCE_STALE')
    x=dict(base); x['baseline_provider']='WRONG'; require_not_ready(v3.infer(x),'BASELINE_PROVIDER_MISMATCH')
    x=dict(base); x['pregame_net_diff']=x['pregame_net_diff']+1; require_not_ready(v3.infer(x),'BASELINE_IDENTITY_MISMATCH')
    x=dict(base); x['source_event_id']='WRONG'; require_not_ready(v3.infer(x),'GAME_IDENTITY_MISMATCH')
    results['negative_canaries']=['MISSING_REQUIRED_FEATURES','WINDOW_MISMATCH_PERIOD','WINDOW_MISMATCH_CLOCK','STALE_LIVE_STATE','TEAM_IDENTITY_MISMATCH','STATE_IDENTITY_MISMATCH_SCORE','STATE_IDENTITY_MISMATCH_DERIVED','BASELINE_PRIOR_GAMES_IDENTITY_MISMATCH','BASELINE_WINDOW_COUNT_INVALID','BASELINE_PRIOR_COUNT_INVALID','BASELINE_SOURCE_STALE','BASELINE_PROVIDER_MISMATCH','BASELINE_IDENTITY_MISMATCH','GAME_IDENTITY_MISMATCH']
    out={'status':'PASS','results':results,'betting_layer':'DISABLED'}
    root=Path('artifacts/runtime_v3_canary');root.mkdir(parents=True,exist_ok=True);(root/'canary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
