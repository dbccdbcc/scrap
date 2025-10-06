

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_future_races_v4_db.py
---------------------------
Same as score_future_races_v4.py, but adds optional MySQL upsert of predictions.

Env (.env or system):
  DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
  MODEL_PATH=outputs/top3_recent_model.json
  FEATS_PATH=outputs/top3_recent_features.json
  ZERO_ODDS=0
  H2H_YEARS=3
  H2H_CACHE=1
  WRITE_DB=1                     # if 1, write predictions to DB
  PRED_TABLE=future_predictions_v4  # table name for outputs
"""

import os, json, warnings, re
import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

# MODEL_PATH = os.getenv("MODEL_PATH", "outputs/top3_recent_model.json")
# FEATS_PATH = os.getenv("FEATS_PATH", "outputs/top3_recent_features.json")
ZERO_ODDS  = int(os.getenv("ZERO_ODDS","1"))
H2H_YEARS  = int(os.getenv("H2H_YEARS","2"))
H2H_CACHE  = int(os.getenv("H2H_CACHE","1"))
WRITE_DB   = int(os.getenv("WRITE_DB","1"))
PRED_TABLE = os.getenv("PRED_TABLE","future_predictions_v4")

use_ranker = True   # 換成 False 就會用 Classifier

if use_ranker:
    MODEL_PATH = os.getenv("MODEL_PATH", "outputs/top3_rank_model.json")
    FEATS_PATH = os.getenv("FEATS_PATH", "outputs/top3_rank_features.json")
    print("➡️ 使用 Ranker 模式")
else:
    MODEL_PATH = os.getenv("MODEL_PATH", "outputs/top3_recent_model.json")
    FEATS_PATH = os.getenv("FEATS_PATH", "outputs/top3_recent_features.json")
    print("➡️ 使用 Classifier 模式")

# ----------------- Helpers (same as v4) -----------------

def normalize_horse_name(name: str) -> str:
    """Normalize horse name: remove trailing brackets and force uppercase."""
    if not isinstance(name, str):
        return name
    s = str(name).strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    return s.upper()

def parse_finish_pos(s):
    try:
        return float(str(s).strip())
    except:
        return np.nan

def parse_lbw(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    st = str(s).strip()
    if st in ("", "-"):
        return 0.0
    total = 0.0
    if "-" in st:
        a, b = st.split("-", 1)
        try: total += float(a)
        except: pass
        st = b
    if "/" in st:
        try:
            num, den = st.split("/", 1)
            total += float(num) / float(den)
        except:
            pass
        return total
    try:
        return float(st)
    except:
        return np.nan

def running_pos_to_cols(rp):
    if rp is None or (isinstance(rp, float) and np.isnan(rp)):
        return (np.nan, np.nan, np.nan)
    parts = str(rp).replace(",", " ").split()
    def to_num(x): 
        try: return float(x)
        except: return np.nan
    early = to_num(parts[0]) if len(parts)>0 else np.nan
    mid   = to_num(parts[1]) if len(parts)>1 else np.nan
    late  = to_num(parts[2]) if len(parts)>2 else np.nan
    return early, mid, late

def make_event_time(df):
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["race_no_int"] = pd.to_numeric(df["race_no"], errors="coerce").fillna(0).astype(int)
    df["event_time"] = df["race_date"] + pd.to_timedelta(df["race_no_int"], unit="s")
    return df

def asof_merge(base, state, key_cols, value_cols):
    base = base.sort_values("event_time")
    state = state.sort_values("event_time")
    out = pd.merge_asof(base, state, on="event_time", by=key_cols, direction="backward")
    for c in value_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    return out

def build_last3(df_hist):
    sub = df_hist[["horse","event_time","pla_num","lbw_num"]].copy().sort_values(["horse","event_time"])
    sub["pla_num_shift"] = sub.groupby("horse")["pla_num"].shift(1)
    sub["lbw_num_shift"] = sub.groupby("horse")["lbw_num"].shift(1)

    roll_pla_mean = sub.groupby("horse")["pla_num_shift"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_pla_min  = sub.groupby("horse")["pla_num_shift"].rolling(3, min_periods=1).min().reset_index(level=0, drop=True)
    roll_lbw_mean = sub.groupby("horse")["lbw_num_shift"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    def slope_last3(x):
        x = x.dropna().values
        if len(x) < 3: return 0.0
        p1, p2, p3 = x[-3:]
        return (p1 - p3)/2.0

    roll_trend = sub.groupby("horse")["pla_num_shift"].rolling(3, min_periods=1).apply(slope_last3, raw=False).reset_index(level=0, drop=True)

    feats = sub[["horse","event_time"]].copy()
    feats["avg_finish_pos_last3"] = roll_pla_mean.fillna(99)
    feats["best_finish_pos_last3"] = roll_pla_min.fillna(99)
    feats["avg_lbw_last3"] = roll_lbw_mean.fillna(9.9)
    feats["trend_finish_pos_last3"] = roll_trend.fillna(0.0)
    return feats.sort_values(["horse","event_time"])

def build_running_pos(df_hist):
    rp = df_hist[["horse","event_time","running_position"]].copy().sort_values(["horse","event_time"])
    rp[["rp_early","rp_mid","rp_late"]] = rp["running_position"].apply(lambda s: pd.Series(running_pos_to_cols(s)))
    for c in ["rp_early","rp_mid","rp_late"]:
        rp[c] = rp.groupby("horse")[c].shift(1)
    out = rp[["horse","event_time"]].copy()
    for c in ["rp_early","rp_mid","rp_late"]:
        out[c+"_mean5"] = rp.groupby("horse")[c].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    out["pace_front_tendency"] = -out["rp_early_mean5"]
    return out.sort_values(["horse","event_time"])

def build_form_state(df_hist, entity_col, n_window, prefix):
    sub = df_hist[[entity_col, "event_time", "top3"]].dropna().sort_values([entity_col,"event_time"]).copy()
    sub["prior"] = sub.groupby(entity_col)["top3"].shift(1)
    sub["cum_runs"] = sub.groupby(entity_col).cumcount()
    sub["roll_sum"] = sub.groupby(entity_col)["prior"].rolling(n_window, min_periods=1).sum().reset_index(level=0, drop=True)
    sub["roll_cnt"] = sub.groupby(entity_col)["prior"].rolling(n_window, min_periods=1).count().reset_index(level=0, drop=True)
    sub["roll_rate"] = (sub["roll_sum"] / sub["roll_cnt"].replace(0, np.nan)).fillna(0.0)
    sub["last_time"] = sub.groupby(entity_col)["event_time"].shift(1)
    out = sub[[entity_col, "event_time"]].copy()
    out[f"{prefix}_prev_runs"] = sub["cum_runs"]
    out[f"{prefix}_top3_rate_{n_window}"] = sub["roll_rate"]
    out[f"{prefix}_days_since"] = (sub["event_time"] - sub["last_time"]).dt.total_seconds().fillna(9999)/86400.0
    out[f"{prefix}_days_since"] = out[f"{prefix}_days_since"].clip(0, 9999)
    return out.sort_values([entity_col,"event_time"])

# ---- H2H cache-aware utilities ----

def build_h2h_hist_pairs(df_recent):
    base = df_recent[["race_date","race_no","event_time","horse","pla_num"]].dropna(subset=["horse","pla_num"]).copy()
    base["race_key"] = base["race_date"].astype(str) + "#" + base["race_no"].astype(str)
    base = base.sort_values(["race_key","pla_num"])
    left  = base[["race_key","event_time","horse","pla_num"]].rename(columns={"horse":"horse_i","pla_num":"pi"})
    right = base[["race_key","event_time","horse","pla_num"]].rename(columns={"horse":"horse_j","pla_num":"pj"})
    pairs = left.merge(right, on=["race_key","event_time"], how="inner")
    pairs = pairs[pairs["horse_i"] != pairs["horse_j"]]
    pairs["win_ij"] = (pairs["pi"] < pairs["pj"]).astype(float)
    pairs = pairs.sort_values(["horse_i","horse_j","event_time"])
    pairs["h2h_rate"] = pairs.groupby(["horse_i","horse_j"])["win_ij"].expanding().mean().reset_index(level=[0,1], drop=True)
    return pairs[["horse_i","horse_j","event_time","h2h_rate"]]

def compute_h2h_for_runners(runners, hist_pairs):
    cur = runners.merge(runners, on=["race_key","event_time"], suffixes=("_i","_j"))
    cur = cur[cur["horse_i"] != cur["horse_j"]]
    cur = pd.merge_asof(cur.sort_values("event_time"),
                        hist_pairs.sort_values("event_time"),
                        on="event_time",
                        by=["horse_i","horse_j"],
                        direction="backward")
    h2h_mean = cur.groupby(["race_key","event_time","horse_i"])["h2h_rate"].mean().reset_index()
    h2h_mean = h2h_mean.rename(columns={"horse_i":"horse","h2h_rate":"h2h_avg_vs_field"})
    h2h_mean["h2h_avg_vs_field"] = h2h_mean["h2h_avg_vs_field"].fillna(0.5)
    return h2h_mean[["race_key","horse","event_time","h2h_avg_vs_field"]]

def load_or_build_h2h_cache(df_hist, df_future):
    cache_parquet = "outputs/h2h_hist_pairs.parquet"
    cache_csv     = "outputs/h2h_hist_pairs.csv"

    hist_pairs = None
    if os.path.exists(cache_parquet):
        try:
            hist_pairs = pd.read_parquet(cache_parquet)
            # ensure proper dtypes
            if 'event_time' in hist_pairs.columns:
                hist_pairs['event_time'] = pd.to_datetime(hist_pairs['event_time'], errors='coerce')
            for col in ['horse_i','horse_j']:
                if col in hist_pairs.columns:
                    hist_pairs[col] = hist_pairs[col].astype(str)
            print(f"📦 Loaded H2H hist_pairs cache (parquet): {len(hist_pairs):,} rows")
        except Exception as e:
            print("⚠️ Failed to load parquet cache:", e)

    if hist_pairs is None and os.path.exists(cache_csv):
        try:
            hist_pairs = pd.read_csv(cache_csv)
            # ensure proper dtypes
            if 'event_time' in hist_pairs.columns:
                hist_pairs['event_time'] = pd.to_datetime(hist_pairs['event_time'], errors='coerce')
            for col in ['horse_i','horse_j']:
                if col in hist_pairs.columns:
                    hist_pairs[col] = hist_pairs[col].astype(str)
            print(f"📦 Loaded H2H hist_pairs cache (csv): {len(hist_pairs):,} rows")
        except Exception as e:
            print("⚠️ Failed to load csv cache:", e)

    if hist_pairs is None:
        latest_future_date = pd.to_datetime(df_future["race_date"].max())
        cutoff = latest_future_date - pd.DateOffset(years=H2H_YEARS)
        df_recent = df_hist[df_hist["race_date"] >= cutoff].copy()
        print(f"🤝 Building H2H historical pairs (recent window {cutoff.date()} → {latest_future_date.date()}) ...")
        hist_pairs = build_h2h_hist_pairs(df_recent)
        print(f"✅ H2H hist_pairs built: {len(hist_pairs):,} rows")
        # ensure dtype
        if 'event_time' in hist_pairs.columns:
            hist_pairs['event_time'] = pd.to_datetime(hist_pairs['event_time'], errors='coerce')
        if H2H_CACHE:
            os.makedirs("outputs", exist_ok=True)
            try:
                hist_pairs.to_parquet(cache_parquet, index=False)
                print(f"💾 Saved H2H cache → {cache_parquet}")
            except Exception as e:
                print("⚠️ Failed to save parquet cache:", e)
                hist_pairs.to_csv(cache_csv, index=False)
                print(f"💾 Saved fallback H2H cache → {cache_csv}")
    return hist_pairs

# ----------------- Main -----------------


def _df_to_mysql_records(df, cols):
    """Convert DataFrame to a list of Python-native dicts without NaN/NaT/Inf.
    Ensures ints/floats are native Python types and missing -> None."""
    import numpy as np
    import pandas as pd
    import math

    # ensure column order and subset
    df2 = df[cols].copy()

    # make nullable ints explicit for rank/pred
    if "pred_top3" in df2.columns:
        df2["pred_top3"] = pd.to_numeric(df2["pred_top3"], errors="coerce").astype("Int64")
    if "rank_in_race" in df2.columns:
        df2["rank_in_race"] = pd.to_numeric(df2["rank_in_race"], errors="coerce").astype("Int64")

    records = []
    for rec in df2.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if v is None:
                clean[k] = None
                continue
            # pandas NA / numpy NaN
            try:
                if pd.isna(v):
                    clean[k] = None
                    continue
            except Exception:
                pass
            # numpy types -> Python native
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                if np.isfinite(v):
                    clean[k] = float(v)
                else:
                    clean[k] = None
            elif hasattr(v, "to_pydatetime"):
                clean[k] = v.to_pydatetime()
            else:
                clean[k] = v
        records.append(clean)
    return records


def _is_finite_number(x):
    try:
        import numpy as np, math
        if isinstance(x, (int,)):
            return True
        if isinstance(x, float):
            return math.isfinite(x)
        if hasattr(x, "dtype") and str(x.dtype).startswith(("float","int")):
            return np.isfinite(x)
    except Exception:
        pass
    return False

def _sanitize_records_strict(df, cols, float_cols=None, int_cols=None, str_cols=None, date_cols=None):
    """Return list of python-native dicts with all NaN/NaT/Inf -> None.
       Also validates no float('nan') left. Raises with example if found."""
    import numpy as np
    import pandas as pd
    import math
    float_cols = float_cols or []
    int_cols = int_cols or []
    str_cols = str_cols or []
    date_cols = date_cols or []

    df2 = df[cols].copy()

    # enforce dtypes softly
    for c in float_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    for c in int_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").astype("Int64")
    for c in date_cols:
        if c in df2.columns:
            df2[c] = pd.to_datetime(df2[c], errors="coerce").dt.date
    for c in str_cols:
        if c in df2.columns:
            df2[c] = df2[c].astype(object).where(df2[c].notna(), None)

    recs = []
    bad_examples = []
    for rec in df2.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            # treat pandas NA / numpy NaN
            try:
                import pandas as pd
                if pd.isna(v):
                    clean[k] = None
                    continue
            except Exception:
                pass
            # numpy types -> Python native
            try:
                import numpy as np, datetime as dt, math
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                elif hasattr(v, "to_pydatetime"):
                    v = v.to_pydatetime()
            except Exception:
                pass
            # strings 'nan' -> None
            if isinstance(v, str) and v.lower() == "nan":
                v = None
            # final numeric check
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                v = None
            clean[k] = v
        # audit for remaining NaN-like
        for k, v in clean.items():
            if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                bad_examples.append(clean)
                break
        recs.append(clean)

    if bad_examples:
        ex = bad_examples[0]
        raise RuntimeError(f"Sanitization failed; found NaN/Inf in record example: {ex}")
    return recs


def main():
    load_dotenv()
    DB_USER = os.getenv("DB_USER"); DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST","localhost"); DB_NAME = os.getenv("DB_NAME")
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise SystemExit("❌ Missing DB credentials in env/.env")

    # clf = xgb.XGBClassifier()
    # if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATS_PATH):
    #     raise SystemExit("❌ Model/features JSON not found. Train v4 first.")
    # clf.load_model(MODEL_PATH)
    # with open(FEATS_PATH,"r",encoding="utf-8") as f:
    #     feature_names = json.load(f).get("feature_names", [])
    from xgboost import XGBClassifier, XGBRanker
    with open(MODEL_PATH, "r", encoding="utf-8") as _f:
        _js = _f.read()
    if ("rank:pairwise" in _js) or ('"learning_task_type":"ranking"' in _js) or ('"rank:' in _js):
        clf = XGBRanker()
    else:
        clf = XGBClassifier()
    clf.load_model(MODEL_PATH)

    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?charset=utf8mb4",
                           poolclass=NullPool)

    # Load history
    sql_hist = text("""
    SELECT id, race_date, course, race_no, race_info, pla, horse_no, horse,
           jockey, trainer, act_wt, declared_wt, draw_no, lbw, running_position,
           win_odds, race_class, distance, surface, track, going
    FROM race_results
    WHERE race_date IS NOT NULL
    """)
    hist = pd.read_sql(sql_hist, engine)
    hist["pla_num"] = hist["pla"].apply(parse_finish_pos)
    hist["lbw_num"] = hist["lbw"].apply(parse_lbw)
    hist = make_event_time(hist)
    hist["top3"] = (hist["pla_num"] <= 3).astype(int)
    for c in ["race_no","distance","horse_no","act_wt","declared_wt","draw_no","win_odds"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    for c in ["course","track","race_class","surface","going"]:
        hist[f"{c}_raw"] = hist[c].astype(str).where(hist[c].notna(), "UNK")

    # Load future entries
    sql_future = text("""
    SELECT id, race_info, horse_no, horse, draw_no, act_wt, jockey, trainer, win_odds, place_odds,
           race_date, course, race_no, race_class, distance, surface, track, going
    FROM future_races
    """)
    fr = pd.read_sql(sql_future, engine)
    fr["horse"] = fr["horse"].apply(normalize_horse_name)  # normalize horse names

    fr = make_event_time(fr)
    for c in ["race_no","distance","horse_no","act_wt","draw_no","win_odds","place_odds"]:
        fr[c] = pd.to_numeric(fr[c], errors="coerce")
    for c in ["course","track","race_class","surface","going"]:
        fr[f"{c}_raw"] = fr[c].astype(str).where(fr[c].notna(), "UNK")
    fr["race_key"] = fr["race_date"].astype(str) + "#" + fr["race_no"].astype(str)

    # H2H (cache-aware)
    hist_pairs = load_or_build_h2h_cache(hist, fr)
    runners = fr[["race_date","race_no","event_time","horse"]].dropna().copy()
    runners["race_key"] = runners["race_date"].astype(str) + "#" + runners["race_no"].astype(str)
    h2h = compute_h2h_for_runners(runners, hist_pairs)

    # Other features from history
    last3 = build_last3(hist)
    rp    = build_running_pos(hist)
    horse_state = build_form_state(hist, "horse", 5, "horse")
    jock_state  = build_form_state(hist, "jockey", 200, "jockey")
    trn_state   = build_form_state(hist, "trainer", 200, "trainer")

    # Draw bias (as-of from history)
    raw_keys = hist[["event_time","course","distance","track","draw_no","top3"]].copy()
    raw_keys["course"]  = raw_keys["course"].astype(str).where(raw_keys["course"].notna(), "UNK")
    raw_keys["track"]   = raw_keys["track"].astype(str).where(raw_keys["track"].notna(), "UNK")
    raw_keys["distance"]= pd.to_numeric(raw_keys["distance"], errors="coerce").fillna(-1).astype("int64")
    raw_keys["draw_no"] = pd.to_numeric(raw_keys["draw_no"], errors="coerce").fillna(-1).astype("int64")
    raw_keys = raw_keys.sort_values(["course","distance","track","event_time","draw_no"])
    g = raw_keys.groupby(["course","distance","track","draw_no"], sort=False)
    prior = g["top3"].shift(1)
    rsum = prior.rolling(200, min_periods=1).sum()
    rcnt = prior.rolling(200, min_periods=1).count()
    rrate = (rsum/rcnt.replace(0,np.nan)).fillna(0.0)
    raw_keys["draw_bias_rate"] = rrate
    raw_keys = raw_keys[["course","distance","track","event_time","draw_no","draw_bias_rate"]]

    # Merge features onto future entries
    base = fr.copy()
    base = asof_merge(base, last3, ["horse"], ["avg_finish_pos_last3","best_finish_pos_last3","avg_lbw_last3","trend_finish_pos_last3"])
    base = base.merge(h2h, on=["race_key","horse","event_time"], how="left")
    base["h2h_avg_vs_field"] = base["h2h_avg_vs_field"].fillna(0.5)
    base = asof_merge(base, rp, ["horse"], ["rp_early_mean5","rp_mid_mean5","rp_late_mean5","pace_front_tendency"])
    base = asof_merge(base, horse_state, ["horse"], ["horse_prev_runs","horse_top3_rate_5","horse_days_since"])
    base = asof_merge(base, jock_state, ["jockey"], ["jockey_prev_runs","jockey_top3_rate_200","jockey_days_since"])
    base = asof_merge(base, trn_state, ["trainer"], ["trainer_prev_runs","trainer_top3_rate_200","trainer_days_since"])

    for c in ["distance","draw_no"]:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(-1).astype("int64")

    base = pd.merge_asof(
        base.sort_values("event_time"),
        raw_keys.sort_values("event_time"),
        left_on="event_time", right_on="event_time",
        left_by=["course_raw","distance","track_raw","draw_no"],
        right_by=["course","distance","track","draw_no"],
        direction="backward"
    )
    base["draw_bias_rate"] = base["draw_bias_rate"].fillna(0.0)

    # Build matrix & align
    keep_obj = {"horse","jockey","trainer","race_info"}
    Xmat = base.copy()
    for c in ["course_raw","race_class_raw","surface_raw","track_raw","going_raw"]:
        if c not in Xmat.columns: Xmat[c] = "UNK"
    Xmat = pd.get_dummies(Xmat, columns=["course_raw","race_class_raw","surface_raw","track_raw","going_raw"], drop_first=True)

    NON_FEATURES = {"id","race_date","race_no","horse_no","horse","jockey","trainer","race_info",
                    "event_time","race_key"}
    datetime_cols = [c for c in Xmat.columns if np.issubdtype(Xmat[c].dtype, np.datetime64) or np.issubdtype(Xmat[c].dtype, np.timedelta64)]
    NON_FEATURES.update(datetime_cols)

    X_full = Xmat[[c for c in Xmat.columns if c not in NON_FEATURES]].copy()

    with open(FEATS_PATH,"r",encoding="utf-8") as f:
        feature_names = json.load(f).get("feature_names", [])
    for f in feature_names:
        if f not in X_full.columns:
            X_full[f] = 0.0
    X_full = X_full[feature_names]

    ODDS_KEYS = ("win_odds","place_odds","implied_prob","log_odds","odds_rank")
    for col in X_full.columns:
        if X_full[col].dtype == 'bool':
            X_full[col] = X_full[col].astype(np.float32)
        elif X_full[col].dtype.kind in 'iu':
            X_full[col] = X_full[col].astype(np.float32)
        elif X_full[col].dtype.kind == 'f':
            X_full[col] = X_full[col].astype(np.float32)
        if ZERO_ODDS and any(k in col for k in ODDS_KEYS):
            X_full[col] = 0.0

    # Predict
    # proba = clf.predict_proba(X_full)[:,1]
    # base["proba_top3_raw"] = proba
    # tmp = base[["race_date","race_no"]].copy()
    # tmp["p"] = proba
    # tmp["p_norm"] = tmp.groupby(["race_date","race_no"])["p"].transform(lambda s: (s/(s.sum()+1e-12)*3).clip(0,1))
    # base["proba_top3"] = tmp["p_norm"].values
    # base["rank_in_race"] = base.groupby(["race_date","race_no"])["proba_top3"].rank(ascending=False, method="first")
    # base["pred_top3"] = (base["rank_in_race"] <= 3).astype(int)
    try:
        # 如果係 classifier → 有 predict_proba
        proba_vec = clf.predict_proba(X_full)[:, 1]
        # 用 logit 轉成 raw 分數（避免 0/1 過於極端）
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_scores = np.log(np.clip(proba_vec, 1e-9, 1-1e-9) / (1 - np.clip(proba_vec, 1e-9, 1-1e-9)))
    except AttributeError:
        # Ranker 無 predict_proba → 用 predict 分數
        raw_scores = clf.predict(X_full)

    # Softmax normalization by race
    base = base.copy()
    base["__score__"] = raw_scores
    proba = np.zeros(len(base), dtype=float)
    for rk, idx in base.groupby("race_key").groups.items():
        idx = list(idx)
        s = base.loc[idx, "__score__"].astype(float).values
        s = s - np.max(s)  # stability
        e = np.exp(s)
        denom = e.sum() if e.sum() > 0 else 1.0
        proba[idx] = e / denom
    
    # Save back
    base["proba_top3"] = proba
    base["rank_in_race"] = base.groupby("race_key")["proba_top3"].rank(ascending=False, method="first").astype(int)
    base["pred_top3"] = (base["rank_in_race"] <= 3).astype(int)

    # Save CSV
    os.makedirs("outputs", exist_ok=True)
    out_cols = ["id","race_date","course_raw","race_no","horse_no","horse","jockey","trainer",
                "draw_no","act_wt","distance","race_class_raw","surface_raw","track_raw","going_raw",
                "win_odds","place_odds","proba_top3","pred_top3","rank_in_race"]
    for c in out_cols:
        if c not in base.columns: base[c] = np.nan
    out = base[out_cols].sort_values(["race_date","race_no","rank_in_race","proba_top3"], ascending=[True,True,True,False])
    out_path = "outputs/future_predictions_ranked_v4.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved → {out_path}")

    # Optional: write to DB with upsert on id
    if WRITE_DB:
        print(f"🗄  Writing predictions to MySQL table `{PRED_TABLE}` (upsert by id)...")
        # Create table if not exists
        ddl = f"""
        CREATE TABLE IF NOT EXISTS `{PRED_TABLE}` (
          `id` INT NOT NULL,
          `race_date` DATE NULL,
          `course_raw` VARCHAR(50) NULL,
          `race_no` INT NULL,
          `horse_no` INT NULL,
          `horse` VARCHAR(100) NULL,
          `jockey` VARCHAR(100) NULL,
          `trainer` VARCHAR(100) NULL,
          `draw_no` INT NULL,
          `act_wt` INT NULL,
          `distance` INT NULL,
          `race_class_raw` VARCHAR(100) NULL,
          `surface_raw` VARCHAR(100) NULL,
          `track_raw` VARCHAR(100) NULL,
          `going_raw` VARCHAR(100) NULL,
          `win_odds` FLOAT NULL,
          `place_odds` FLOAT NULL,
          `proba_top3` FLOAT NULL,
          `pred_top3` TINYINT NULL,
          `rank_in_race` INT NULL,
          `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
        """
        with engine.begin() as conn:
            conn.execute(text(ddl))

        # Prepare rows & upsert
        rows = out.copy()
        rows["race_date"] = pd.to_datetime(rows["race_date"], errors="coerce").dt.date
        cols = ["id","race_date","course_raw","race_no","horse_no","horse","jockey","trainer",
                "draw_no","act_wt","distance","race_class_raw","surface_raw","track_raw","going_raw",
                "win_odds","place_odds","proba_top3","pred_top3","rank_in_race"]

        # Convert NaN to None so MySQL gets NULL instead of NaN
        rows = rows[cols].replace({np.nan: None})
        # Ensure types (ints may be floats after processing)
        for c in ["id","race_no","horse_no","draw_no","distance","pred_top3","rank_in_race"]:
            if c in rows.columns:
                rows[c] = rows[c].apply(lambda v: int(v) if v is not None and pd.notna(v) else None)
        for c in ["act_wt","win_odds","place_odds","proba_top3"]:
            if c in rows.columns:
                rows[c] = rows[c].apply(lambda v: float(v) if v is not None and pd.notna(v) else None)

        # Build parameterized UPSERT
        insert_cols = ", ".join(f"`{c}`" for c in cols)
        values_cols = ", ".join(f"%({c})s" for c in cols)  
        update_cols = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in cols[1:])  
        sql = f"""
            INSERT INTO `{PRED_TABLE}` ({insert_cols})
            VALUES ({values_cols})
            ON DUPLICATE KEY UPDATE {update_cols}, `updated_at`=CURRENT_TIMESTAMP
        """  


        # Build parameterized UPSERT
        float_cols = ["win_odds","place_odds","proba_top3"]
        int_cols   = ["id","race_no","horse_no","draw_no","act_wt","distance","pred_top3","rank_in_race"]
        str_cols   = ["course_raw","horse","jockey","trainer","race_class_raw","surface_raw","track_raw","going_raw"]
        date_cols  = ["race_date"]
        records = _sanitize_records_strict(rows, cols, float_cols, int_cols, str_cols, date_cols)

        # Chunked execute
        CHUNK = 1000
        with engine.begin() as conn:
            for i in range(0, len(records), CHUNK):
                conn.exec_driver_sql(sql, records[i:i+CHUNK])
        print(f"✅ Upserted {len(records):,} rows into `{PRED_TABLE}`")

if __name__ == "__main__":
    main()
