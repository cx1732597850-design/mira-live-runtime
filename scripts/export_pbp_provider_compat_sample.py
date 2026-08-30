import json
from pathlib import Path
import pandas as pd
import trend_runtime_v2 as v2


def main():
    df=v2.download([2025])
    cols=list(df.columns)
    string_cols=[c for c in cols if df[c].dtype=='object']
    phoenix_ids=set()
    matches=[]
    for c in string_cols:
        s=df[c].astype(str)
        m=s.str.contains('Phoenix|Mercury',case=False,na=False)
        if m.any():
            matches.append({'column':c,'values':s[m].drop_duplicates().head(10).tolist()})
            for idc in ['offense_team_id','defense_team_id']:
                if idc in df.columns: phoenix_ids.update(df.loc[m,idc].dropna().astype(str).tolist())
    date_cols=[c for c in cols if 'date' in c.lower() or 'time' in c.lower()]
    out={'columns':cols,'string_matches':matches,'candidate_phoenix_ids':sorted(phoenix_ids),'date_cols':date_cols}
    # summarize final 8 games for any detected IDs, using exact PBP team totals
    samples=[]
    for tid in sorted(phoenix_ids):
        try: tv=float(tid) if '.' in tid else int(tid)
        except: tv=tid
        gids=df[(df.offense_team_id==tv)|(df.defense_team_id==tv)].game_id.drop_duplicates().tolist()
        for gid in gids[-8:]:
            g=df[df.game_id==gid]; teams=sorted(set(g.offense_team_id.unique()).union(set(g.defense_team_id.unique())))
            if tv not in teams or len(teams)!=2: continue
            opp=teams[0] if teams[1]==tv else teams[1]
            t=v2.team_tot(g,tv); o=v2.team_tot(g,opp); poss=(t['poss']+o['poss'])/2.0
            rec={'game_id':str(gid),'team_id':str(tid),'opp_id':str(opp),'pts':t['score'],'opp_pts':o['score'],'poss':poss,'ortg':100*t['score']/poss,'drtg':100*o['score']/poss,'pace':poss}
            for c in date_cols[:4]:
                vals=g[c].dropna().astype(str).drop_duplicates().head(2).tolist();
                if vals: rec[c]=vals
            samples.append(rec)
    out['samples']=samples[-12:]
    Path('artifacts/provider_compat_sample').mkdir(parents=True,exist_ok=True)
    Path('artifacts/provider_compat_sample/sample.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
