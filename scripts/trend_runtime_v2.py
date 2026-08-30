import json, math, os, sys, hashlib, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

SOURCES={
2023:("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2023.parquet","9e9a7d04ec3f62ac51337f65e6a3038265577ad96b0e7f093ec2e4fda4a1df38"),
2024:("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2024.parquet","db7bf27b64c962ed0311d74b423e107f62dd25fcbca007bf872919f78f84ce45"),
2025:("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2025.parquet","bb3870acb35a2e5bcbe5adda5037e8b7b09797e6ad9265a96efad11773067ec0"),
2026:("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2026.parquet","5d17cfab7da2228a9bf5677661e570fd2fd0a66a5ec7df9648d1dbe3b9e64a8e")}
STAT_COLS=['fg2a','fg2m','fg3a','fg3m','fta','ftm','oreb','dreb','tov']
MODES={'Q1_2P5':([1],150.0),'Q1_END':([1,2],600.0),'Q2_2P5':([1,2],750.0)}
BASELINE_COLS=['pregame_net_diff','pregame_pace_diff','pregame_off_diff','pregame_def_diff','pregame_games_min']


def download(years):
    root=Path('artifacts/runtime_v2/source'); root.mkdir(parents=True,exist_ok=True)
    out=[]
    for y in years:
        url,sha=SOURCES[y]; p=root/f'wnba_possessions_{y}.parquet'
        if not p.exists(): urllib.request.urlretrieve(url,p)
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=sha: raise RuntimeError(f'hash mismatch {y}: {actual}')
        d=pd.read_parquet(p); d['season']=y; out.append(d)
    return pd.concat(out,ignore_index=True)

def snap_rows(g,mode):
    if mode=='Q1_2P5': return g[(g.period==1)&(g.end_seconds_remaining>=450)]
    if mode=='Q1_END': return g[g.period==1]
    if mode=='Q2_2P5': return g[(g.period==1)|((g.period==2)&(g.end_seconds_remaining>=450))]
    raise ValueError(mode)

def team_tot(rows,team):
    r=rows[rows.offense_team_id==team]
    o={'score':float(r.points.sum()),'poss':float(r.count_as_possession.sum())}
    for c in STAT_COLS:o[c]=float(r[c].sum())
    return o

def game_order_key(gid):
    s=''.join(ch for ch in str(gid) if ch.isdigit())
    return int(s) if s else str(gid)

def full_game_team_rows(df,season):
    rows=[]
    for gid,g in df[df.season==season].groupby('game_id',sort=False):
        teams=sorted(set(g.offense_team_id.unique()).union(set(g.defense_team_id.unique())))
        if len(teams)!=2: continue
        a,b=teams; ta,tb=team_tot(g,a),team_tot(g,b)
        poss=max((ta['poss']+tb['poss'])/2.0,1.0)
        for t,opp,x,y in [(a,b,ta,tb),(b,a,tb,ta)]:
            rows.append({'season':season,'game_id':str(gid),'order':game_order_key(gid),'team':str(t),'opp':str(opp),
                         'off_rtg':100*x['score']/poss,'def_rtg':100*y['score']/poss,'net_rtg':100*(x['score']-y['score'])/poss,
                         'pace':poss/40.0*40.0})
    return pd.DataFrame(rows).sort_values(['order','game_id','team']).reset_index(drop=True)

def add_pregame_baseline(games):
    hist={}; out=[]
    for _,r in games.iterrows():
        t=r.team; h=hist.get(t,[])
        if h:
            b={k:float(np.mean([x[k] for x in h[-10:]])) for k in ['off_rtg','def_rtg','net_rtg','pace']}
            n=len(h)
        else:
            b={k:0.0 for k in ['off_rtg','def_rtg','net_rtg','pace']}; n=0
        q=dict(r); q.update({f'pre_{k}':v for k,v in b.items()}); q['pre_games']=n; out.append(q)
        hist.setdefault(t,[]).append(dict(r))
    return pd.DataFrame(out)

def build_states(df):
    baselines={}
    for season in sorted(df.season.unique()):
        gr=add_pregame_baseline(full_game_team_rows(df,season))
        baselines[season]={(r.game_id,r.team):r for _,r in gr.iterrows()}
    rows=[]
    for (season,gid),g in df.groupby(['season','game_id'],sort=False):
        teams=sorted(set(g.offense_team_id.unique()).union(set(g.defense_team_id.unique())))
        if len(teams)!=2: continue
        a,b=map(str,teams)
        for mode,(target_periods,elapsed) in MODES.items():
            s=snap_rows(g,mode)
            if s.empty: continue
            ta,tb=team_tot(s,float(a) if a.replace('.','',1).isdigit() else a),team_tot(s,float(b) if b.replace('.','',1).isdigit() else b)
            # recover exact IDs when numeric conversion is unsafe
            aa,bb=teams; ta,tb=team_tot(s,aa),team_tot(s,bb)
            target=g[g.period.isin(target_periods)]; fa,fb=team_tot(target,aa),team_tot(target,bb)
            ba=baselines[season].get((str(gid),str(aa))); bbline=baselines[season].get((str(gid),str(bb)))
            rec={'season':int(season),'game_id':str(gid),'team_a':str(aa),'team_b':str(bb),'mode':mode}
            for prefix,t in [('a',ta),('b',tb)]:
                for k,v in t.items(): rec[f'{prefix}_{k}']=v
            rec['cur_margin']=ta['score']-tb['score']; rec['cur_total']=ta['score']+tb['score']; rec['poss_total']=ta['poss']+tb['poss']; rec['poss_diff']=ta['poss']-tb['poss']; rec['elapsed_sec']=elapsed; rec['pace_poss_per_min']=rec['poss_total']/(elapsed/60.0)
            rec['target_margin']=fa['score']-fb['score']; rec['target_total']=fa['score']+fb['score']
            if ba is None or bbline is None or ba.pre_games<3 or bbline.pre_games<3:
                rec.update({c:np.nan for c in BASELINE_COLS})
            else:
                rec['pregame_net_diff']=ba.pre_net_rtg-bbline.pre_net_rtg
                rec['pregame_pace_diff']=ba.pre_pace-bbline.pre_pace
                rec['pregame_off_diff']=ba.pre_off_rtg-bbline.pre_off_rtg
                rec['pregame_def_diff']=ba.pre_def_rtg-bbline.pre_def_rtg
                rec['pregame_games_min']=min(ba.pre_games,bbline.pre_games)
            rows.append(rec)
    return pd.DataFrame(rows)

def feature_cols(ds):
    drop={'season','game_id','team_a','team_b','mode','target_margin','target_total'}
    return [c for c in ds.columns if c not in drop]

def fit_models(ds,mode,train_seasons):
    d=ds[(ds['mode']==mode)&ds[BASELINE_COLS].notna().all(axis=1)].copy(); feats=feature_cols(d); models={}
    for target in ['target_margin','target_total']:
        p=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=10.0))]); p.fit(d[d.season.isin(train_seasons)][feats],d[d.season.isin(train_seasons)][target]); models[target]=p
    return models,feats,d

def strength(x):
    a=abs(float(x))
    return 'NO_CLEAR_TREND' if a<3 else ('WEAK_MODERATE_TREND' if a<5 else 'STRONG_TREND')

def verify():
    df=download([2023,2024,2025,2026]); ds=build_states(df); out={'status':'PASS','modes':{},'thresholds':{'no_clear_lt':3.0,'strong_gte':5.0},'2026_baseline_coverage':{}}
    for mode in MODES:
        models,feats,d=fit_models(ds,mode,[2023,2024]); test=d[d.season==2025].copy(); pm=models['target_margin'].predict(test[feats]); pt=models['target_total'].predict(test[feats]); y=test.target_margin.to_numpy(); nz=y!=0
        out['modes'][mode]={'train_n':int((d.season.isin([2023,2024])).sum()),'oos_n':int(len(test)),'margin_mae':float(mean_absolute_error(y,pm)),'margin_rmse':float(mean_squared_error(y,pm)**0.5),'direction_accuracy_ex_ties':float(np.mean(np.sign(pm[nz])==np.sign(y[nz]))),'total_mae':float(mean_absolute_error(test.target_total,pt)),'strong_n':int((np.abs(pm)>=5).sum()),'strong_accuracy_ex_ties':float(np.mean(np.sign(pm[(np.abs(pm)>=5)&nz])==np.sign(y[(np.abs(pm)>=5)&nz]))) if ((np.abs(pm)>=5)&nz).any() else None}
    d26=ds[ds.season==2026]
    for mode in MODES:
        x=d26[d26['mode']==mode]; out['2026_baseline_coverage'][mode]={'rows':int(len(x)),'baseline_ready_rows':int(x[BASELINE_COLS].notna().all(axis=1).sum())}
    Path('artifacts/runtime_v2').mkdir(parents=True,exist_ok=True); ds.to_parquet('artifacts/runtime_v2/state_windows_v2.parquet',index=False); json.dump(out,open('artifacts/runtime_v2/verification.json','w'),indent=2); print(json.dumps(out,indent=2))

def infer(payload):
    required=['mode','cur_margin','cur_total','poss_total','poss_diff','elapsed_sec','pace_poss_per_min','a_score','a_poss','b_score','b_poss','pregame_net_diff','pregame_pace_diff','pregame_off_diff','pregame_def_diff','pregame_games_min']
    missing=[k for k in required if k not in payload or payload[k] is None]
    if missing: return {'status':'NOT_READY','missing':missing}
    mode=payload['mode']
    if mode not in MODES: return {'status':'NOT_READY','reason':'INVALID_MODE'}
    if float(payload['pregame_games_min'])<3: return {'status':'NOT_READY','reason':'INSUFFICIENT_2026_PREGAME_BASELINE'}
    df=download([2023,2024,2025]); ds=build_states(df); models,feats,d=fit_models(ds,mode,[2023,2024,2025])
    row={c:0.0 for c in feats}
    for c in feats:
        if c in payload: row[c]=payload[c]
    x=pd.DataFrame([row])[feats]; pm=float(models['target_margin'].predict(x)[0]); pt=float(models['target_total'].predict(x)[0])
    return {'status':'TREND_FINAL','mode':mode,'pred_margin':pm,'pred_total':pt,'direction':'TEAM_A' if pm>0 else ('TEAM_B' if pm<0 else 'TIE'),'trend_strength':strength(pm),'betting_layer':'DISABLED'}

if __name__=='__main__':
    if len(sys.argv)>=2 and sys.argv[1]=='verify': verify()
    elif len(sys.argv)>=3 and sys.argv[1]=='infer': print(json.dumps(infer(json.load(open(sys.argv[2]))),indent=2))
    else: print('usage: trend_runtime_v2.py verify | infer payload.json')
