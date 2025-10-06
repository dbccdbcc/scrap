#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_moving_cv_v4_features_fast.py
-----------------------------------
Moving-window CV with enhanced features, optimized for speed:

Speed-ups
- Vectorized head-to-head (H2H) with a single merge_asof
- Optional caching of historical pair expanding means to Parquet
- Limit history window for H2H to recent years (default 5y)

Features (all as-of)
- Last-3 form (finish pos & beaten length)
- Head-to-head vs field (historical dominance)
- Running-position pace proxies (shifted history)
- Rolling entity states (horse/jockey/trainer)
- Draw bias

Env (.env or system):
  DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
  START_DATE=2010-01-01
  END_DATE=2025-08-01
  USE_ODDS=0/1        # 1 use odd
  H2H_YEARS=5         # only use last N years of history for H2H (and draw bias calc)
  H2H_CACHE=1         # cache hist expanding means to outputs/h2h_hist_pairs.parquet
"""

import os, json, warnings
import re

def normalize_horse(s):
    try:
        s = str(s).strip()
    except Exception:
        return s
    # Remove trailing ' (XXXXXX...)' tokens if present (e.g., 'BRAVE HEART (CK277)' -> 'BRAVE HEART')
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    return s.upper()

from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

TARGET = "top3"
USE_ODDS   = int(os.getenv("USE_ODDS", "1"))
START_DATE = os.getenv("START_DATE", "2021-09-01")
END_DATE   = os.getenv("END_DATE",   "2026-08-01")
H2H_YEARS  = int(os.getenv("H2H_YEARS", "2"))
H2H_CACHE  = int(os.getenv("H2H_CACHE", "1"))
RANK_MODE  = int(os.getenv("RANK_MODE", "1"))

XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="auc",
    tree_method="hist",
    n_jobs=0,
    random_state=42
)

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
            total += float(num)/float(den)
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

def add_last3_features(df):
    sub = df[["horse","event_time","pla_num","lbw_num"]].copy().sort_values(["horse","event_time"])
    sub["pla_num_shift"] = sub.groupby("horse")["pla_num"].shift(1)
    sub["lbw_num_shift"] = sub.groupby("horse")["lbw_num"].shift(1)

    roll_pla_mean = sub.groupby("horse")["pla_num_shift"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_pla_min  = sub.groupby("horse")["pla_num_shift"].rolling(3, min_periods=1).min().reset_index(level=0, drop=True)
    roll_lbw_mean = sub.groupby("horse")["lbw_num_shift"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    def slope_last3(x):
        x = x.dropna().values
        if len(x) < 3:
            return 0.0
        p1, p2, p3 = x[-3:]
        return (p1 - p3) / 2.0
    roll_trend = sub.groupby("horse")["pla_num_shift"].rolling(3, min_periods=1).apply(slope_last3, raw=False).reset_index(level=0, drop=True)

    feats = sub[["horse","event_time"]].copy()
    feats["avg_finish_pos_last3"] = roll_pla_mean.fillna(99)
    feats["best_finish_pos_last3"] = roll_pla_min.fillna(99)
    feats["avg_lbw_last3"] = roll_lbw_mean.fillna(9.9)
    feats["trend_finish_pos_last3"] = roll_trend.fillna(0.0)
    return feats.sort_values(["horse","event_time"])

def build_running_pos(df):
    rp = df[["horse","event_time","running_position"]].copy().sort_values(["horse","event_time"])
    rp[["rp_early","rp_mid","rp_late"]] = rp["running_position"].apply(lambda s: pd.Series(running_pos_to_cols(s)))
    for c in ["rp_early","rp_mid","rp_late"]:
        rp[c] = rp.groupby("horse")[c].shift(1)
    out = rp[["horse","event_time"]].copy()
    for c in ["rp_early","rp_mid","rp_late"]:
        out[c+"_mean5"] = rp.groupby("horse")[c].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    out["pace_front_tendency"] = -out["rp_early_mean5"]
    return out.sort_values(["horse","event_time"])

def build_form_state(df_hist, entity_col, n_window, prefix):
    sub = df_hist[[entity_col, "event_time", TARGET]].dropna().sort_values([entity_col,"event_time"]).copy()
    sub["prior"] = sub.groupby(entity_col)[TARGET].shift(1)
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

def build_h2h_hist_pairs(df_recent):
    """
    Step 1: Build historical ordered pairs (i,j) with expanding mean h2h_rate.
    Returns DataFrame ['horse_i','horse_j','event_time','h2h_rate']
    """
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
    """
    Step 2/3: For given runners (race_key,event_time,horse), build all directed pairs (i,j),
    single merge_asof to fetch h2h_rate, then average per (race, horse_i).
    """
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

def main():
    load_dotenv()
    DB_USER = os.getenv("DB_USER"); DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST","localhost"); DB_NAME = os.getenv("DB_NAME")
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
        raise SystemExit("❌ Missing DB credentials in env/.env")

    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}?charset=utf8mb4",
                           poolclass=NullPool)

    print("🔌 Connecting and loading data...")
    sql = text("""
    SELECT id, race_date, course, race_no, race_info, pla, horse_no, horse,
           jockey, trainer, act_wt, declared_wt, draw_no, lbw, running_position,
           win_odds, race_class, distance, surface, track, going
    FROM race_results
    WHERE race_date BETWEEN :sd AND :ed
    """)
    df = pd.read_sql(sql, engine, params={"sd": START_DATE, "ed": END_DATE})
    df["horse"] = df["horse"].astype(str).map(normalize_horse)
    print(f"Loaded rows: {len(df):,}")

    df = make_event_time(df)
    df["pla_num"] = df["pla"].apply(parse_finish_pos)
    df["lbw_num"] = df["lbw"].apply(parse_lbw)
    df[TARGET] = (df["pla_num"] <= 3).astype(int)

    for c in ["course","track","race_class","surface","going"]:
        df[f"{c}_raw"] = df[c].astype(str).where(df[c].notna(), "UNK")
    for c in ["race_no","distance","horse_no","act_wt","declared_wt","draw_no","win_odds"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ------- Feature blocks (fast) -------
    print("🧱 Building last-3 form features...")
    last3 = add_last3_features(df)

    print("🏃 Building running-position pace proxies...")
    rp = build_running_pos(df)

    # Limit recent years for H2H & draw bias
    cutoff_date = pd.to_datetime(END_DATE) - pd.DateOffset(years=H2H_YEARS)
    df_recent = df[df["race_date"] >= cutoff_date].copy()
    print(f"🕒 H2H/DrawBias history window: from {cutoff_date.date()} to {pd.to_datetime(END_DATE).date()} (rows={len(df_recent):,})")

    
    # ---- H2H with caching (load if exists, else build and save) ----
    cache_path = "outputs/h2h_hist_pairs.parquet"
    hist_pairs = None
    if H2H_CACHE and os.path.exists(cache_path):
        try:
            hist_pairs = pd.read_parquet(cache_path)
            # ensure expected dtypes
            if "event_time" in hist_pairs.columns:
                hist_pairs["event_time"] = pd.to_datetime(hist_pairs["event_time"], errors="coerce")
            for col in ("horse_i","horse_j"):
                if col in hist_pairs.columns:
                    hist_pairs[col] = hist_pairs[col].astype(str)
            print(f"📦 Loaded H2H hist_pairs cache → {cache_path} ({len(hist_pairs):,} rows)")
        except Exception as e:
            print("⚠️ Failed to load H2H cache:", e)
            hist_pairs = None

    if hist_pairs is None:
        print("🤝 Building H2H historical pairs (vectorized)...")
        hist_pairs = build_h2h_hist_pairs(df_recent)
        print(f"✅ H2H hist_pairs built: {len(hist_pairs):,} rows")
        if H2H_CACHE:
            os.makedirs("outputs", exist_ok=True)
            try:
                hist_pairs.to_parquet(cache_path, index=False)
                print(f"💾 Saved H2H hist_pairs cache → {cache_path}")
            except Exception as e:
                print("⚠️ Failed to save H2H cache:", e)
                # optional CSV fallback
                try:
                    hist_pairs.to_csv("outputs/h2h_hist_pairs.csv", index=False)
                    print("💾 Saved fallback H2H cache → outputs/h2h_hist_pairs.csv")
                except Exception as e2:
                    print("⚠️ Failed to save fallback H2H cache:", e2)

    print("📊 Aggregating H2H vs field for all races...")
    runners = df_recent[["race_date","race_no","event_time","horse"]].dropna().copy()
    runners["race_key"] = runners["race_date"].astype(str) + "#" + runners["race_no"].astype(str)
    h2h = compute_h2h_for_runners(runners, hist_pairs)
    print(f"✅ H2H aggregated: {len(h2h):,} rows")

    # Draw bias from recent
    print("🎯 Computing draw bias (recent years)...")
    raw_keys = df_recent[["event_time","course_raw","distance","track_raw","going_raw","draw_no",TARGET]].copy()
    for c in ["distance","draw_no"]:
        raw_keys[c] = pd.to_numeric(raw_keys[c], errors="coerce").fillna(-1).astype("int64")
    raw_keys = raw_keys.sort_values(["course_raw","distance","track_raw","going_raw","event_time","draw_no"])
    g = raw_keys.groupby(["course_raw","distance","track_raw","going_raw","draw_no"], sort=False)
    prior = g[TARGET].shift(1)
    rsum = prior.rolling(200, min_periods=1).sum()
    rcnt = prior.rolling(200, min_periods=1).count()
    rrate = (rsum/rcnt.replace(0,np.nan)).fillna(0.0)
    raw_keys["draw_bias_rate"] = rrate
    raw_keys = raw_keys[["course_raw","distance","track_raw","going_raw","event_time","draw_no","draw_bias_rate"]]

    # Merge all as-of features
    print("🔗 Merging features...")
    base = df.copy()
    base["race_key"] = base["race_date"].astype(str) + "#" + base["race_no"].astype(str)

    base = asof_merge(base, last3, ["horse"], ["avg_finish_pos_last3","best_finish_pos_last3","avg_lbw_last3","trend_finish_pos_last3"])
    base = base.merge(h2h, on=["race_key","horse","event_time"], how="left")
    base["h2h_avg_vs_field"] = base["h2h_avg_vs_field"].fillna(0.5)
    base = asof_merge(base, rp, ["horse"], ["rp_early_mean5","rp_mid_mean5","rp_late_mean5","pace_front_tendency"])

    # entity states
    def build_form_state_local(ent, w, pfx):
        return build_form_state(base, ent, w, pfx)
    base = asof_merge(base, build_form_state_local("horse",5,"horse"), ["horse"], ["horse_prev_runs","horse_top3_rate_5","horse_days_since"])
    base = asof_merge(base, build_form_state_local("jockey",200,"jockey"), ["jockey"], ["jockey_prev_runs","jockey_top3_rate_200","jockey_days_since"])
    base = asof_merge(base, build_form_state_local("trainer",200,"trainer"), ["trainer"], ["trainer_prev_runs","trainer_top3_rate_200","trainer_days_since"])

    for c in ["distance", "draw_no"]:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(-1).astype("int64")

    base = pd.merge_asof(
        base.sort_values("event_time"),
        raw_keys.sort_values("event_time"),
        on="event_time",
        by=["course_raw","distance","track_raw","going_raw","draw_no"],
        direction="backward"
        )
    base["draw_bias_rate"] = base["draw_bias_rate"].fillna(0.0)

    # Build matrix
    print("🧮 Building feature matrix...")
    keep_obj = {"horse","jockey","trainer","race_info","pla","running_position"}
    Xy = base.copy()

    if not USE_ODDS:
        Xy = Xy.drop(columns=[c for c in ["win_odds"] if c in Xy.columns])

    drop_obj = [c for c in Xy.columns if Xy[c].dtype=='O' and (not c.endswith("_raw")) and c not in keep_obj]
    Xy = Xy.drop(columns=drop_obj)

    for c in ["course_raw","race_class_raw","surface_raw","track_raw","going_raw"]:
        if c not in Xy.columns: Xy[c] = "UNK"
    Xy = pd.get_dummies(Xy, columns=["course_raw","race_class_raw","surface_raw","track_raw","going_raw"], drop_first=True)

    NON_FEATURES = {"id","race_date","race_no","horse_no","horse","jockey","trainer","race_info","pla",
                    "running_position","event_time",TARGET,"pla_num","lbw_num"}
    datetime_cols = [c for c in Xy.columns if np.issubdtype(Xy[c].dtype, np.datetime64) or np.issubdtype(Xy[c].dtype, np.timedelta64)]
    NON_FEATURES.update(datetime_cols)

    y = Xy[TARGET].astype(int)
    X = Xy[[c for c in Xy.columns if c not in NON_FEATURES]].copy()

    for c in X.columns:
        if X[c].dtype == 'bool':
            X[c] = X[c].astype(np.float32)
        elif X[c].dtype.kind in 'iu':
            X[c] = X[c].astype(np.float32)
        elif X[c].dtype.kind == 'f':
            X[c] = X[c].astype(np.float32)

    # Moving-window CV (monthly 3-month val windows)
    print("🚦 Starting moving-window CV...")
    dates = pd.to_datetime(base["race_date"].dropna().unique())
    months = sorted({pd.Timestamp(d.year, d.month, 1) for d in dates})
    VAL_MONTHS = 3
    TRAIN_YEARS_MIN = 3
    folds = []
    for i in range(TRAIN_YEARS_MIN*12, len(months)-VAL_MONTHS):
        train_end = months[i]
        val_start = months[i]
        val_end = months[i+VAL_MONTHS]
        folds.append((train_end, val_start, val_end))

    print(f"Total folds: {len(folds)}")
    metrics = []
    feature_names = list(X.columns)

    for idx, (train_end, val_start, val_end) in enumerate(tqdm(folds, desc="CV folds")):
        m_train = (base["race_date"] < train_end)
        m_val   = (base["race_date"] >= val_start) & (base["race_date"] < val_end)

        X_tr, y_tr = X.loc[m_train], y.loc[m_train]
        X_va, y_va = X.loc[m_val], y.loc[m_val]
        if len(X_tr)==0 or len(X_va)==0:
            continue

        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(X_tr, y_tr)

        va_proba = clf.predict_proba(X_va)[:,1]
        va_pred = (va_proba >= 0.5).astype(int)

        tp = int(((va_pred==1) & (y_va==1)).sum())
        fp = int(((va_pred==1) & (y_va==0)).sum())
        fn = int(((va_pred==0) & (y_va==1)).sum())
        precision = tp / (tp+fp) if (tp+fp)>0 else 0.0
        recall    = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

        metrics.append({
            "fold": idx+1,
            "train_end": str(train_end.date()),
            "val_start": str(val_start.date()),
            "val_end": str(val_end.date()),
            "precision": precision, "recall": recall, "f1": f1,
            "train_rows": int(m_train.sum()), "val_rows": int(m_val.sum())
        })

    met = pd.DataFrame(metrics)
    print("CV summary (means):", met[["precision","recall","f1"]].mean().to_dict())

    # Retrain on all data and save
    clf_final = xgb.XGBClassifier(**XGB_PARAMS)
    clf_final.fit(X, y)
    os.makedirs("outputs", exist_ok=True)
    
    if RANK_MODE:
        # Train an XGBRanker using race_key grouping (pairwise ranking)
        from xgboost import XGBRanker
        # Reorder X, y by race_key to create contiguous groups
        rk = base["race_key"].astype(str).values
        order = np.argsort(rk, kind="stable")
        X_rank = X.iloc[order].reset_index(drop=True)
        y_rank = y.iloc[order].reset_index(drop=True)
        rk_sorted = rk[order]
        # Build group sizes in contiguous order
        group_sizes = []
        last = None; cnt = 0
        for v in rk_sorted:
            if v != last and last is not None:
                group_sizes.append(cnt); cnt = 1; last = v
            else:
                cnt = cnt + 1 if last is not None else 1; last = v
        if cnt: group_sizes.append(cnt)
        # Reasonable params for ranking
        ranker = XGBRanker(
            objective="rank:pairwise",
            eval_metric="ndcg@3",
            n_estimators=XGB_PARAMS.get("n_estimators", 800),
            max_depth=XGB_PARAMS.get("max_depth", 6),
            learning_rate=XGB_PARAMS.get("learning_rate", 0.05),
            subsample=XGB_PARAMS.get("subsample", 0.9),
            colsample_bytree=XGB_PARAMS.get("colsample_bytree", 0.9),
            tree_method=XGB_PARAMS.get("tree_method", "hist"),
            n_jobs=0,
            random_state=42
        )
        ranker.fit(X_rank, y_rank, group=group_sizes)
        ranker.save_model("outputs/top3_rank_model.json")
        with open("outputs/top3_rank_features.json","w",encoding="utf-8") as f:
            json.dump({"feature_names": list(X_rank.columns)}, f, ensure_ascii=False, indent=2)
        print("✅ Saved ranker model → outputs/top3_rank_model.json")
        print("✅ Saved ranker features → outputs/top3_rank_features.json")
        # Still save the classifier for backward compat
        clf_final.save_model("outputs/top3_recent_model.json")
    # Always save the classifier model as the primary predictor
    clf_final.save_model("outputs/top3_recent_model.json")
    with open("outputs/top3_recent_features.json","w",encoding="utf-8") as f:
        json.dump({"feature_names": feature_names}, f, ensure_ascii=False, indent=2)
    print("✅ Saved model → outputs/top3_recent_model.json")
    print("✅ Saved features → outputs/top3_recent_features.json")

if __name__ == "__main__":
    main()
