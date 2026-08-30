from __future__ import annotations
import hashlib, json, os, sys, urllib.request
from pathlib import Path
import pandas as pd

ASSETS = {
    2023: {
        "url": "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2023.parquet",
        "sha256": "9e9a7d04ec3f62ac51337f65e6a3038265577ad96b0e7f093ec2e4fda4a1df38",
    },
    2024: {
        "url": "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2024.parquet",
        "sha256": "db7bf27b64c962ed0311d74b423e107f62dd25fcbca007bf872919f78f84ce45",
    },
    2025: {
        "url": "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/wnba_stats_possessions/wnba_possessions_2025.parquet",
        "sha256": "bb3870acb35a2e5bcbe5adda5037e8b7b09797e6ad9265a96efad11773067ec0",
    },
}

ROOT = Path("artifacts/materialized")
ROOT.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, path: Path):
    req = urllib.request.Request(url, headers={"User-Agent": "mira-live-runtime/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, path.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

def main():
    manifest = {
        "pipeline": "MIRA-WNBA-LIVE-V1.0-corpus-materialization",
        "seasons": {},
        "source_release": "sportsdataverse/sportsdataverse-data@wnba_stats_possessions",
        "status": "STARTED",
    }
    schemas = {}
    for season, spec in ASSETS.items():
        path = ROOT / f"wnba_possessions_{season}.parquet"
        download(spec["url"], path)
        got = sha256_file(path)
        if got != spec["sha256"]:
            raise RuntimeError(f"SHA256 mismatch {season}: {got} != {spec['sha256']}")
        df = pd.read_parquet(path)
        schema = {str(c): str(df[c].dtype) for c in df.columns}
        schemas[str(season)] = {
            "columns": list(map(str, df.columns)),
            "dtypes": schema,
            "rows": int(len(df)),
            "duplicate_full_rows": int(df.duplicated().sum()),
            "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
        }
        sample_path = ROOT / f"sample_{season}.jsonl"
        df.head(25).to_json(sample_path, orient="records", lines=True, date_format="iso")
        manifest["seasons"][str(season)] = {
            "url": spec["url"],
            "expected_sha256": spec["sha256"],
            "actual_sha256": got,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "duplicate_full_rows": int(df.duplicated().sum()),
            "file_bytes": int(path.stat().st_size),
        }
    # cross-season schema audit
    colsets = {s: set(v["columns"]) for s, v in schemas.items()}
    common = sorted(set.intersection(*colsets.values())) if colsets else []
    union = sorted(set.union(*colsets.values())) if colsets else []
    manifest["schema_audit"] = {
        "common_columns": common,
        "union_columns": union,
        "identical_column_sets": len({tuple(sorted(x)) for x in colsets.values()}) == 1,
    }
    manifest["status"] = "MATERIALIZED_HASH_VERIFIED_SCHEMA_AUDITED"
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ROOT / "schemas.json").write_text(json.dumps(schemas, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
