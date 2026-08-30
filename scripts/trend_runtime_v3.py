import json, math
from pathlib import Path
from datetime import datetime

BUNDLE_PATH=Path(__file__).resolve().parents[1]/'runtime'/'runtime_v3_linear_bundle.json'
STAT_KEYS=['fg2a','fg2m','fg3a','fg3m','fta','ftm','oreb','dreb','tov']
BASELINE_KEYS=['pregame_net_diff','pregame_pace_diff','pregame_off_diff','pregame_def_diff','pregame_games_min']
META_KEYS=['game_id','team_a_id','team_b_id','mode','period','clock_seconds_remaining','captured_at_et','execution_time_et','input_source','source_event_id','baseline_provider','baseline_asof_et','baseline_source_id','baseline_source_hash','baseline_games_a','baseline_games_b']
VALID_MODES={'Q1_2P5','Q1_END','Q2_2P5'}
BASELINE_PROVIDER='STATMUSE_WNBA_GAME_LEVEL_ADVANCED'


def load_bundle():
    b=json.loads(BUNDLE_PATH.read_text(encoding='utf-8'))
    if b.get('version')!='MIRA-WNBA-LIVE-RUNTIME-V3-LINEAR': raise RuntimeError('BUNDLE_IDENTITY_FAIL')
    return b


def fail(reason,**kw):
    out={'status':'NOT_READY','reason':reason,'betting_layer':'DISABLED'}; out.update(kw); return out


def parse_dt(x):
    try:
        d=datetime.fromisoformat(str(x).replace('Z','+00:00'))
        if d.tzinfo is None: return None
        return d
    except Exception: return None


def normalise_team(p,prefix):
    q={}
    for k in ['score','poss','fg2a','fg2m','fg3a','fg3m','fta','ftm','oreb','dreb','tov','fgm','fga']:
        key=f'{prefix}_{k}'
        if key in p and p[key] is not None: q[k]=p[key]
    if ('fg2a' not in q or 'fg2m' not in q) and all(k in q for k in ['fga','fgm','fg3a','fg3m']):
        q['fg2a']=float(q['fga'])-float(q['fg3a']); q['fg2m']=float(q['fgm'])-float(q['fg3m'])
    needed=['score','fg2a','fg2m','fg3a','fg3m','fta','ftm','oreb','dreb','tov']
    miss=[f'{prefix}_{k}' for k in needed if k not in q]
    if miss: return None,miss
    try: q={k:float(v) for k,v in q.items()}
    except Exception: return None,[f'{prefix}_NON_NUMERIC']
    if 'poss' not in q:
        q['poss']=(q['fg2a']+q['fg3a'])+0.44*q['fta']-q['oreb']+q['tov']
        q['poss_method']='BOX_ESTIMATE_044'
    else: q['poss_method']='SOURCE_EXACT'
    return q,[]


def validate_team(q,prefix):
    nums=['score','poss','fg2a','fg2m','fg3a','fg3m','fta','ftm','oreb','dreb','tov']
    if any((not math.isfinite(float(q[k])) or float(q[k])<0) for k in nums): return f'{prefix}_NEGATIVE_OR_NONFINITE'
    if q['fg2m']>q['fg2a'] or q['fg3m']>q['fg3a'] or q['ftm']>q['fta']: return f'{prefix}_MAKES_GT_ATTEMPTS'
    expected=2*q['fg2m']+3*q['fg3m']+q['ftm']
    if abs(expected-q['score'])>1e-9: return f'{prefix}_SCORE_STAT_MISMATCH'
    return None


def window_check(mode,period,clock):
    if mode=='Q1_2P5': return period==1 and 420<=clock<=480
    if mode=='Q1_END': return period==1 and abs(clock)<=1e-9
    if mode=='Q2_2P5': return period==2 and 420<=clock<=480
    return False


def elapsed(mode,clock):
    if mode=='Q1_2P5': return 600-clock
    if mode=='Q1_END': return 600.0
    return 600+(600-clock)


def strength(x):
    a=abs(float(x))
    return 'NO_CLEAR_TREND' if a<3 else ('WEAK_MODERATE_TREND' if a<5 else 'STRONG_TREND')


def dot(model,features,row):
    return float(model['b']+sum(float(w)*float(row[f]) for w,f in zip(model['w'],features)))


def infer(payload):
    missing=[k for k in META_KEYS+BASELINE_KEYS if k not in payload or payload[k] is None]
    if missing: return fail('MISSING_REQUIRED_FIELDS',missing=missing)
    if not str(payload['input_source']).strip(): return fail('INPUT_SOURCE_INVALID')
    mode=str(payload['mode'])
    if mode not in VALID_MODES: return fail('INVALID_MODE')
    if str(payload['team_a_id'])==str(payload['team_b_id']): return fail('TEAM_IDENTITY_MISMATCH')
    if str(payload['source_event_id'])!=str(payload['game_id']): return fail('GAME_IDENTITY_MISMATCH')
    try: period=int(payload['period']); clock=float(payload['clock_seconds_remaining'])
    except Exception: return fail('INVALID_PERIOD_CLOCK')
    if not window_check(mode,period,clock): return fail('WINDOW_MISMATCH',mode=mode,period=period,clock_seconds_remaining=clock)
    cap=parse_dt(payload['captured_at_et']); exe=parse_dt(payload['execution_time_et']); base=parse_dt(payload['baseline_asof_et'])
    if cap is None or exe is None or base is None: return fail('INVALID_TIMESTAMP')
    age=(exe-cap).total_seconds()
    if age < -5: return fail('LIVE_STATE_FROM_FUTURE',age_seconds=age)
    if age > 120: return fail('STALE_LIVE_STATE',age_seconds=age)
    if base>cap: return fail('BASELINE_AFTER_LIVE_STATE')
    baseline_age=(cap-base).total_seconds()
    if baseline_age>36*3600: return fail('BASELINE_SOURCE_STALE',baseline_age_hours=baseline_age/3600)
    if str(payload['baseline_provider'])!=BASELINE_PROVIDER: return fail('BASELINE_PROVIDER_MISMATCH',required=BASELINE_PROVIDER)
    h=str(payload['baseline_source_hash']).strip().lower()
    if not str(payload['baseline_source_id']).strip() or len(h)!=64 or any(c not in '0123456789abcdef' for c in h): return fail('BASELINE_PROVENANCE_INVALID')
    try:
        ga=int(payload['baseline_games_a']); gb=int(payload['baseline_games_b'])
        if not (3<=ga<=10 and 3<=gb<=10): return fail('BASELINE_SAMPLE_COUNT_INVALID')
        for k in BASELINE_KEYS:
            if not math.isfinite(float(payload[k])): return fail('BASELINE_NONFINITE',field=k)
        if int(float(payload['pregame_games_min']))!=min(ga,gb): return fail('BASELINE_SAMPLE_IDENTITY_MISMATCH')
        if float(payload['pregame_games_min'])<3: return fail('INSUFFICIENT_2026_PREGAME_BASELINE')
        # Because NetRtg is defined row-wise as ORtg-DRtg and arithmetic means are used, the differences must obey this identity.
        if abs(float(payload['pregame_net_diff'])-(float(payload['pregame_off_diff'])-float(payload['pregame_def_diff'])))>1e-6:
            return fail('BASELINE_IDENTITY_MISMATCH')
    except ValueError: return fail('BASELINE_NON_NUMERIC')
    except TypeError: return fail('BASELINE_NON_NUMERIC')
    a,ma=normalise_team(payload,'a'); b,mb=normalise_team(payload,'b')
    if ma or mb: return fail('MISSING_REQUIRED_FEATURES',missing=ma+mb)
    ea=validate_team(a,'A'); eb=validate_team(b,'B')
    if ea or eb: return fail('STATE_IDENTITY_MISMATCH',detail=ea or eb)
    if a['poss']<0.25 or b['poss']<0.25: return fail('STATE_IDENTITY_MISMATCH',detail='IMPOSSIBLE_POSSESSIONS')
    if abs(a['poss']-b['poss'])>3.0: return fail('STATE_IDENTITY_MISMATCH',detail='POSSESSION_IMBALANCE')
    el=elapsed(mode,clock)
    row={}
    for pref,q in [('a',a),('b',b)]:
        row[f'{pref}_score']=q['score']; row[f'{pref}_poss']=q['poss']
        for k in STAT_KEYS: row[f'{pref}_{k}']=q[k]
    row['cur_margin']=a['score']-b['score']; row['cur_total']=a['score']+b['score']; row['poss_total']=a['poss']+b['poss']; row['poss_diff']=a['poss']-b['poss']; row['elapsed_sec']=el; row['pace_poss_per_min']=row['poss_total']/(el/60.0)
    for k in BASELINE_KEYS: row[k]=float(payload[k])
    for k in ['cur_margin','cur_total','poss_total','poss_diff','elapsed_sec','pace_poss_per_min']:
        if k in payload and payload[k] is not None:
            try:
                if abs(float(payload[k])-row[k])>1e-6: return fail('STATE_IDENTITY_MISMATCH',detail=f'DERIVED_{k}_MISMATCH')
            except Exception: return fail('STATE_IDENTITY_MISMATCH',detail=f'DERIVED_{k}_NON_NUMERIC')
    bundle=load_bundle(); features=bundle['features']
    miss=[f for f in features if f not in row]
    if miss: return fail('MISSING_MODEL_FEATURES',missing=miss)
    m=bundle['modes'][mode]; pm=dot(m['margin'],features,row); pt=dot(m['total'],features,row)
    status='PREVIEW_CHECK' if mode=='Q1_END' else 'TREND_FINAL'
    return {'status':status,'game_id':str(payload['game_id']),'team_a_id':str(payload['team_a_id']),'team_b_id':str(payload['team_b_id']),'mode':mode,'pred_margin':pm,'fair_spread':pm,'pred_total':pt,'direction':'TEAM_A' if pm>0 else ('TEAM_B' if pm<0 else 'TIE'),'trend_strength':strength(pm),'possession_method_a':a['poss_method'],'possession_method_b':b['poss_method'],'input_contract':'PASS','baseline_contract':'PASS','baseline_provider':BASELINE_PROVIDER,'bundle_version':bundle['version'],'bundle_sha256_source':bundle['bundle_sha256_source'],'betting_layer':'DISABLED'}

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2: raise SystemExit('usage: trend_runtime_v3.py payload.json')
    print(json.dumps(infer(json.load(open(sys.argv[1]))),indent=2))
