import os, json, hashlib, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

SOURCES = {
  2023: ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2023.parquet", "9e9a7d04ec3f62ac51337f65e6a3038265577ad96b0e7f093ec2e4fda4a1df38"),
  2024: ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2024.parquet", "db7bf27b64c962ed0311d74b423e107f62dd25fcbca007bf872919f78f84ce45"),
  2025: ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2025.parquet", "bb3870acb35a2e5bcbe5adda5037e8b7b09797e6ad9265a96efad11773067ec0")
}
OUT=Path('artifacts/fit_oos'); OUT.mkdir(parents=True, exist_ok=True)
DATA=Path('artifacts/source'); DATA.mkdir(parents=True, exist_ok=True)
STAT_COLS=['fg2a','fg2m','fg3a','fg3m','fta','ftm','oreb','dreb','tov']

def download():
    frames=[]
    manifest={}
    for y,(url,sha) in SOURCES.items():
        p=DATA/f'wnba_possessions_{y}.parquet'
        if not p.exists(): urllib.request.urlretrieve(url,p)
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=sha: raise RuntimeError(f'hash mismatch {y}: {actual}')
        df=pd.read_parquet(p)
        frames.append(df)
        manifest[str(y)]={'sha256':actual,'rows':int(len(df)),'columns':int(df.shape[1])}
    return pd.concat(frames,ignore_index=True), manifest

def snap_rows(g, mode):
    if mode=='Q1_2P5': return g[(g.period==1)&(g.end_seconds_remaining>=450)]
    if mode=='Q1_END': return g[g.period==1]
    if mode=='Q2_2P5': return g[(g.period==1)|((g.period==2)&(g.end_seconds_remaining>=450))]
    raise ValueError(mode)

def totals_by_team(rows, team):
    r=rows[rows.offense_team_id==team]
    out={'score':float(r.points.sum()),'poss':float(r.count_as_possession.sum())}
    for c in STAT_COLS: out[c]=float(r[c].sum())
    return out

def make_record(g, season, mode, target_periods):
    teams=sorted(set(g.offense_team_id.unique()).union(set(g.defense_team_id.unique())))
    if len(teams)!=2: return None
    a,b=teams
    s=snap_rows(g,mode)
    if s.empty: return None
    ta,tb=totals_by_team(s,a),totals_by_team(s,b)
    target=g[g.period.isin(target_periods)]
    fa,fb=totals_by_team(target,a),totals_by_team(target,b)
    rec={'season':int(season),'game_id':str(g.game_id.iloc[0]),'team_a':str(a),'team_b':str(b),'mode':mode}
    for prefix,t in [('a',ta),('b',tb)]:
        for k,v in t.items(): rec[f'{prefix}_{k}']=v
    rec['cur_margin']=ta['score']-tb['score']; rec['cur_total']=ta['score']+tb['score']
    rec['poss_total']=ta['poss']+tb['poss']; rec['poss_diff']=ta['poss']-tb['poss']
    if mode=='Q1_2P5': elapsed=150.0
    elif mode=='Q1_END': elapsed=600.0
    else: elapsed=750.0
    rec['elapsed_sec']=elapsed
    rec['pace_poss_per_min']=rec['poss_total']/(elapsed/60.0)
    rec['target_margin']=fa['score']-fb['score']; rec['target_total']=fa['score']+fb['score']
    rec['remaining_margin']=rec['target_margin']-rec['cur_margin']; rec['remaining_total']=rec['target_total']-rec['cur_total']
    return rec

def build_states(df):
    rows=[]
    for (season,gid),g in df.groupby(['season','game_id'],sort=False):
        for mode,target_periods in [('Q1_2P5',[1]),('Q1_END',[1,2]),('Q2_2P5',[1,2])]:
            r=make_record(g,season,mode,target_periods)
            if r: rows.append(r)
    return pd.DataFrame(rows)

def fit_eval(ds, mode):
    d=ds[ds['mode']==mode].copy()
    train=d[d.season.isin([2023,2024])].copy(); test=d[d.season==2025].copy()
    drop={'season','game_id','mode','target_margin','target_total','remaining_margin','remaining_total'}
    feats=[c for c in d.columns if c not in drop]
    cats=['team_a','team_b']; nums=[c for c in feats if c not in cats]
    prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num',StandardScaler(),nums)])
    metrics={'mode':mode,'train_n':int(len(train)),'oos_n':int(len(test)),'features':feats,'markets_available':False}
    preds={}
    for target in ['target_margin','target_total']:
        pipe=Pipeline([('prep',prep),('model',Ridge(alpha=10.0))])
        pipe.fit(train[feats],train[target])
        p=pipe.predict(test[feats]); y=test[target].to_numpy()
        metrics[target]={
          'mae':float(mean_absolute_error(y,p)),
          'rmse':float(mean_squared_error(y,p)**0.5),
          'mean_actual':float(np.mean(y)),
          'mean_pred':float(np.mean(p))
        }
        if target=='target_margin':
            nz=y!=0
            metrics[target]['winner_direction_accuracy_ex_ties']=float(np.mean(np.sign(p[nz])==np.sign(y[nz]))) if nz.any() else None
        preds[target]=p
    out=test[['season','game_id','team_a','team_b','mode','cur_margin','cur_total','target_margin','target_total']].copy()
    out['pred_margin']=preds['target_margin']; out['pred_total']=preds['target_total']
    return metrics,out

def main():
    df,src=download()
    states=build_states(df)
    states.to_parquet(OUT/'state_windows.parquet',index=False)
    states.to_csv(OUT/'state_windows.csv',index=False)
    results={'pipeline':'MIRA-WNBA-LIVE-V1.0-1Q-1H-fit-oos','train_seasons':[2023,2024],'untouched_oos_season':2025,'source':src,'state_counts':states.groupby(['season','mode']).size().unstack(fill_value=0).to_dict(),'status':'PREDICTIVE_STATE_OOS_COMPLETE_MARKET_VALIDATION_PENDING','market_validation_blocker':'Possession corpus contains no historical synchronized 1Q/1H live spread/total lines or prices. OOS here validates outcome prediction only; it cannot validate edge, price gate, betting calibration, or P&L.'}
    allpred=[]
    for mode in ['Q1_2P5','Q1_END','Q2_2P5']:
        m,p=fit_eval(states,mode); results[mode]=m; allpred.append(p)
    pd.concat(allpred).to_csv(OUT/'oos_predictions_2025.csv',index=False)
    with open(OUT/'oos_report.json','w') as f: json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
