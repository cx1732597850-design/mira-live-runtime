import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import trend_runtime_v2 as v2


def est_poss(r,p):
    return float(r[f'{p}_fg2a']+r[f'{p}_fg3a']+0.44*r[f'{p}_fta']-r[f'{p}_oreb']+r[f'{p}_tov'])

def main():
    df=v2.download([2023,2024,2025]); ds=v2.build_states(df); report={'status':'PASS','train_seasons':[2023,2024],'untouched_oos':2025,'modes':{}}
    for mode in v2.MODES:
        models,feats,d=v2.fit_models(ds,mode,[2023,2024]); t=d[d.season==2025].copy(); exact=t[feats].copy(); approx=exact.copy()
        for idx,r in t.iterrows():
            ap=est_poss(r,'a'); bp=est_poss(r,'b'); el=float(r.elapsed_sec)
            approx.loc[idx,'a_poss']=ap; approx.loc[idx,'b_poss']=bp; approx.loc[idx,'poss_total']=ap+bp; approx.loc[idx,'poss_diff']=ap-bp; approx.loc[idx,'pace_poss_per_min']=(ap+bp)/(el/60.0)
        pe=models['target_margin'].predict(exact); pa=models['target_margin'].predict(approx); te=models['target_total'].predict(exact); ta=models['target_total'].predict(approx); y=t.target_margin.to_numpy(); yt=t.target_total.to_numpy(); nz=y!=0
        sx=np.abs(pa)>=5
        report['modes'][mode]={
          'n':int(len(t)),
          'exact_direction_accuracy':float(np.mean(np.sign(pe[nz])==np.sign(y[nz]))),
          'screenshot_direction_accuracy':float(np.mean(np.sign(pa[nz])==np.sign(y[nz]))),
          'direction_flip_rate_vs_exact':float(np.mean(np.sign(pa)!=np.sign(pe))),
          'exact_margin_mae':float(mean_absolute_error(y,pe)),
          'screenshot_margin_mae':float(mean_absolute_error(y,pa)),
          'exact_total_mae':float(mean_absolute_error(yt,te)),
          'screenshot_total_mae':float(mean_absolute_error(yt,ta)),
          'screenshot_strong_n':int(sx.sum()),
          'screenshot_strong_accuracy_ex_ties':float(np.mean(np.sign(pa[sx & nz])==np.sign(y[sx & nz]))) if (sx & nz).any() else None,
          'mean_abs_pred_margin_shift':float(np.mean(np.abs(pa-pe))),
          'mean_abs_pred_total_shift':float(np.mean(np.abs(ta-te)))
        }
    root=Path('artifacts/screenshot_adapter_oos');root.mkdir(parents=True,exist_ok=True);(root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
