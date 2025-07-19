import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from tqdm import tqdm
import re

def clean_horse_name(name):
    if not isinstance(name, str):
        return ''
    return re.sub(r'\s*\(.*?\)', '', name).strip()

# ---------- DB Setup ----------
tqdm.pandas()
print("Loading environment variables and connecting to the database...")
load_dotenv()
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

engine = create_engine(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}?charset=utf8mb4"
)

# ---------- Load Data ----------
racedate_count = pd.read_sql("SELECT COUNT(*) AS cnt FROM racedates", engine).iloc[0, 0]
print(f"racedates table contains {racedate_count} rows.")

print("Reading recent race_results from the database...")
if racedate_count < 400:
    query = """
    SELECT * FROM race_results
    WHERE raceDateId >= (
        SELECT MIN(id) FROM racedates
    )
    """
else:
    query = """
    SELECT * FROM race_results
    WHERE raceDateId >= (
        SELECT MAX(id) - 399 FROM racedates
    )
    """
race_results = pd.read_sql(query, engine)
print(f"race_results loaded: {race_results.shape[0]} rows.")

print("Reading future_races table from the database...")
future_races = pd.read_sql("SELECT * FROM future_races", engine)
print(f"future_races loaded: {future_races.shape[0]} rows.")

# ---------- Clean Horse Names for Both Tables ----------
race_results['horse_clean'] = race_results['horse'].apply(clean_horse_name)
future_races['horse_clean'] = future_races['horse'].apply(clean_horse_name)

# ---------- Process Labels ----------
print("Processing labels...")
def parse_pla(x):
    try:
        return int(x)
    except:
        return np.nan
race_results['pla_num'] = race_results['pla'].apply(parse_pla)
race_results['placed'] = race_results['pla_num'].isin([1,2,3]).astype(int)
race_results['won'] = race_results['pla_num'].eq(1).astype(int)

# ---------- Feature Engineering for Historical Races ----------
print("Feature engineering for historical races (looping, please wait)...")
def get_last_features(horse_clean, date):
    prev = race_results[(race_results['horse_clean'] == horse_clean) & (race_results['race_date'] < date)]
    prev = prev.sort_values('race_date', ascending=False)
    last10 = prev.head(10)
    out = {}

    # Basic stats from last 10 runs
    out['runs'] = len(prev)
    out['avg_place10'] = last10['pla_num'].mean() if len(last10) > 0 else np.nan
    out['avg_odds10'] = pd.to_numeric(last10['win_odds'], errors='coerce').mean() if len(last10) > 0 else np.nan
    out['avg_draw10'] = pd.to_numeric(last10['draw_no'], errors='coerce').mean() if 'draw_no' in last10 and len(last10) > 0 else np.nan
    out['avg_act_wt10'] = pd.to_numeric(last10['act_wt'], errors='coerce').mean() if 'act_wt' in last10 and len(last10) > 0 else np.nan
    out['avg_declared_wt10'] = pd.to_numeric(last10['declared_wt'], errors='coerce').mean() if 'declared_wt' in last10 and len(last10) > 0 else np.nan

    if not last10.empty:
        last = last10.iloc[0]
        out['last_place'] = last['pla_num']
        out['last_odds'] = pd.to_numeric(last['win_odds'], errors='coerce')
        out['last_draw_no'] = pd.to_numeric(last.get('draw_no', np.nan), errors='coerce')
        out['last_act_wt'] = pd.to_numeric(last.get('act_wt', np.nan), errors='coerce')
        out['last_declared_wt'] = pd.to_numeric(last.get('declared_wt', np.nan), errors='coerce')
        out['last_lbw'] = pd.to_numeric(last.get('lbw', np.nan), errors='coerce')
        out['last_finish_time'] = pd.to_numeric(last.get('finish_time', np.nan), errors='coerce')
        out['days_since_last'] = (pd.to_datetime(date) - pd.to_datetime(last['race_date'])).days
        out['last_race_class'] = last.get('race_class', None)
        out['last_distance'] = last.get('distance', None)
        out['last_surface'] = last.get('surface', None)
        out['last_track'] = last.get('track', None)
        out['last_going'] = last.get('going', None)
        out['last_course'] = last.get('course', None)
        out['last_jockey'] = last.get('jockey', None)
        out['last_trainer'] = last.get('trainer', None)
    else:
        out.update({
            'last_place': np.nan,
            'last_odds': np.nan,
            'last_draw_no': np.nan,
            'last_act_wt': np.nan,
            'last_declared_wt': np.nan,
            'last_lbw': np.nan,
            'last_finish_time': np.nan,
            'days_since_last': np.nan,
            'last_race_class': None,
            'last_distance': None,
            'last_surface': None,
            'last_track': None,
            'last_going': None,
            'last_course': None,
            'last_jockey': None,
            'last_trainer': None,
        })
    # Mode features for categorical fields from last 10
    for col in ['race_class','surface','track','going','course','jockey','trainer']:
        if col in last10 and not last10.empty:
            out[f'mode_{col}10'] = last10[col].mode().iloc[0] if not last10[col].mode().empty else None
        else:
            out[f'mode_{col}10'] = None
    return pd.Series(out)

feature_list = []
for idx, row in tqdm(race_results.iterrows(), total=len(race_results), desc="  Historical Features"):
    feature_list.append(get_last_features(row['horse_clean'], row['race_date']))
features = pd.DataFrame(feature_list)
race_results = pd.concat([race_results.reset_index(drop=True), features], axis=1)

categorical_cols = [
    'last_race_class', 'last_surface', 'last_track', 'last_going', 'last_course', 'last_jockey', 'last_trainer',
    'mode_race_class10', 'mode_surface10', 'mode_track10', 'mode_going10', 'mode_course10', 'mode_jockey10', 'mode_trainer10'
]
race_results = pd.get_dummies(race_results, columns=categorical_cols, dummy_na=True)
race_results = race_results.loc[:, ~race_results.columns.duplicated()]

feature_cols = [
    'runs', 'avg_place10', 'avg_odds10', 'avg_draw10', 'avg_act_wt10', 'avg_declared_wt10',
    'last_place', 'last_odds', 'last_draw_no', 'last_act_wt', 'last_declared_wt', 'last_lbw', 'last_finish_time', 'days_since_last'
]
feature_cols += [c for c in race_results.columns if any(col + '_' in c for col in categorical_cols)]
feature_cols = pd.unique(feature_cols)

train = race_results.dropna(subset=['last_place', 'last_odds', 'placed', 'won'])
print(f"Training data prepared: {train.shape[0]} samples.")

X = train[feature_cols]
y_place = train['placed']
y_win = train['won']

print("Training XGBoost model for place probability...")
place_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
place_model.fit(X, y_place)
print("Place model training complete.")

print("Training XGBoost model for win probability...")
win_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
win_model.fit(X, y_win)
print("Win model training complete.")

print("Feature engineering for future races...")
future_feature_list = []
for idx, row in tqdm(future_races.iterrows(), total=len(future_races), desc="  Future Features"):
    future_feature_list.append(get_last_features(row['horse_clean'], row['race_date']))
future_features = pd.DataFrame(future_feature_list)
future_races = pd.concat([future_races.reset_index(drop=True), future_features], axis=1)

future_races = pd.get_dummies(future_races, columns=categorical_cols, dummy_na=True)
future_races = future_races.loc[:, ~future_races.columns.duplicated()]

for col in feature_cols:
    if col not in future_races.columns:
        future_races[col] = 0
X_future = future_races[feature_cols].fillna(-1)

future_races['place_probability'] = place_model.predict_proba(X_future)[:,1]
future_races['win_probability'] = win_model.predict_proba(X_future)[:,1]

# ---------- ADJUST probabilities for horses with little history ----------
min_runs = 10
mean_place_prob = train['placed'].mean()
mean_win_prob = train['won'].mean()
alpha = np.minimum(future_races['runs'] / min_runs, 1.0)
future_races['place_probability'] = (
    alpha * future_races['place_probability'] +
    (1 - alpha) * mean_place_prob
)
future_races['win_probability'] = (
    alpha * future_races['win_probability'] +
    (1 - alpha) * mean_win_prob
)

# ---------- Advanced Monte Carlo Simulation ----------
def advanced_mc_simulation(df, n_sim=10000):
    """Simulate placings for a single race DataFrame with N horses."""
    n = df.shape[0]
    results = np.zeros((n, 3))  # For 1st, 2nd, 3rd place finish counts
    win_probs = df['win_probability'].values
    if win_probs.sum() == 0:
        win_probs = np.ones(n) / n
    for sim in range(n_sim):
        # 1. Draw winner
        win_idx = np.random.choice(n, p=win_probs/win_probs.sum())
        # 2. Remove winner and draw 2nd
        place_probs = np.delete(df['place_probability'].values, win_idx)
        place_probs = place_probs / place_probs.sum() if place_probs.sum() > 0 else np.ones(n-1) / (n-1)
        sec_idx_in_orig = [i for i in range(n) if i != win_idx]
        sec_idx = np.random.choice(len(place_probs), p=place_probs)
        sec_true_idx = sec_idx_in_orig[sec_idx]
        # 3. Remove both for 3rd
        third_indices = [i for i in range(n) if i not in [win_idx, sec_true_idx]]
        third_probs = np.delete(df['place_probability'].values, [win_idx, sec_true_idx])
        third_probs = third_probs / third_probs.sum() if third_probs.sum() > 0 else np.ones(n-2) / (n-2)
        third_in_third_indices = np.random.choice(len(third_probs), p=third_probs)
        third_true_idx = third_indices[third_in_third_indices]
        # Update counts
        results[win_idx, 0] += 1
        results[sec_true_idx, 1] += 1
        results[third_true_idx, 2] += 1
    results /= n_sim
    return results  # shape: (n_horses, 3)

print("Running advanced Monte Carlo simulations for future races...")
output = []
race_groups = list(future_races.groupby(['race_date', 'race_no']))
for (race_id, race_df) in tqdm(race_groups, desc="Monte Carlo Races"):
    mc_results = advanced_mc_simulation(race_df, n_sim=10000)
    for i, (_, row) in enumerate(race_df.iterrows()):
        mc_top3 = mc_results[i, 0] + mc_results[i, 1] + mc_results[i, 2]
        output.append({
            'race_date': race_id[0],
            'race_no': race_id[1],
            'horse_no': row['horse_no'],
            'horse': row['horse'],
            'win_probability': row['win_probability'],
            'place_probability': row['place_probability'],
            'mc_win_pct': mc_results[i, 0],
            'mc_2nd_pct': mc_results[i, 1],
            'mc_3rd_pct': mc_results[i, 2],
            'mc_top3_pct': mc_top3,
        })
sim_results = pd.DataFrame(output)
print("Monte Carlo simulation complete.")

# ---------- Prepare Final Results Table ----------
final_results = sim_results[['race_date', 'race_no', 'horse_no', 'horse', 'win_probability', 'place_probability', 'mc_win_pct', 'mc_2nd_pct', 'mc_3rd_pct', 'mc_top3_pct']]

print("Saving results to MySQL table 'race_predictions'...")
final_results.to_sql(
    name='race_predictions',
    con=engine,
    if_exists='replace',   # or 'append'
    index=False
)
print("Results saved to MySQL table 'race_predictions'.")

# ---------- Save to CSV (optional) ----------
# final_results.to_csv('race_predictions_output.csv', index=False)
# print("Results saved to race_predictions_output.csv")
