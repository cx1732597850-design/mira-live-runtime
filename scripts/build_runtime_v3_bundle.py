import json, hashlib
from pathlib import Path
import numpy as np
import trend_runtime_v2 as rt


def serialise_pipeline(p):
    scaler=p.named_steps['scale']; ridge=p.named_steps['ridge']
    return {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'coef': ridge.coef_.tolist(),
        'intercept': float(ridge.intercept_),
        'alpha': float(ridge.alpha),
    }

def main():
    df=rt.download([2023,2024,2025])
    ds=rt.build_states(df)
    bundle={'bundle_version':'MIRA-WNBA-LIVE-RUNTIME-V3','train_seasons':[2023,2024,2025],'source_sha256':{str(y):rt.SOURCES[y][1] for y in [2023,2024,2025]},'modes':{}}
    for mode in rt.MODES:
        models,feats,d=rt.fit_models(ds,mode,[2023,2024,2025])
        bundle['modes'][mode]={'features':feats,'train_n':int(len(d[d.season.isin([2023,2024,2025])])),'margin':serialise_pipeline(models['target_margin']),'total':serialise_pipeline(models['target_total'])}
    payload=json.dumps(bundle,sort_keys=True,separators=(',',':'))
    bundle['bundle_sha256']=hashlib.sha256(payload.encode()).hexdigest()
    out=Path('artifacts/runtime_v3_bundle'); out.mkdir(parents=True,exist_ok=True)
    (out/'runtime_v3_bundle.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','bundle_sha256':bundle['bundle_sha256'],'modes':{k:{'train_n':v['train_n'],'feature_count':len(v['features'])} for k,v in bundle['modes'].items()}},indent=2))

if __name__=='__main__': main()
