import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from tqdm import tqdm
import re

# ---------- Horse Name Cleaning Function ----------
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
    query = "SELECT * FROM race_results WHERE raceDateId >= (SELECT MIN(id) FROM racedates)"
else:
    query = "SELECT * FROM race_results WHERE raceDateId >= (SELECT MAX(id) - 399 FROM racedates)"
race_results = pd.read_sql(query, engine)
print(f"race_results loaded: {race_results.shape[0]} rows.")

print("Reading future_races table from the database...")
future_races = pd.read_sql("SELECT * FROM future_races", engine)
print(f"future_races loaded: {future_races.shape[0]} rows.")

# ---------- Clean Horse Names ----------
race_results['horse_clean'] = race_results['horse'].apply(clean_horse_name)
future_races['horse_clean'] = future_races['horse'].apply(clean_horse_name)

# ---------- Process Labels ----------
def parse_pla(x):
    try:
        return int(x)
    except:
        return np.nan
race_results['pla_num'] = race_results['pla'].apply(parse_pla)
race_results['placed'] = race_results['pla_num'].isin([1,2,3]).astype(int)
race_results['won'] = race_results['pla_num'].eq(1).astype(int)

# ---------- Feature Engineering for Historical Races ----------
def safe_mode(series):
    """Returns mode or NaN for empty."""
    return series.mode().iloc[0] if not series.mode().empty else np.nan

def get_last10_features(horse_clean, date):
    prev = race_results[(race_results['horse_clean'] == horse_clean) & (race_results['race_date'] < date)]
    prev = prev.sort_values('race_date', ascending=False)
    last10 = prev.head(10)
    # If not enough history, pad with NaN
    features = {}
    features['avg_place10'] = last10['pla_num'].mean() if not last10.empty else np.nan
    features['runs10'] = len(last10)
    features['days_since_last'] = (pd.to_datetime(date) - pd.to_datetime(last10['race_date']).max()).days if not last10.empty else np.nan
    features['avg_odds10'] = pd.to_numeric(last10['win_odds'], errors='coerce').mean() if not last10.empty else np.nan
    features['mode_course10'] = safe_mode(last10['course'])
    features['mode_jockey10'] = safe_mode(last10['jockey'])
    features['mode_trainer10'] = safe_mode(last10['trainer'])
    features['mode_race_class10'] = safe_mode(last10['race_class'])
    features['mode_surface10'] = safe_mode(last10['surface'])
    features['mode_track10'] = safe_mode(last10['track'])
    features['mode_going10'] = safe_mode(last10['going'])
    features['avg_distance10'] = pd.to_numeric(last10['distance'], errors='coerce').mean() if not last10.empty else np.nan
    return pd.Series(features)

print("Feature engineering for historical races (looping, please wait)...")
feature_list = []
for idx, row in tqdm(race_results.iterrows(), total=len(race_results), desc="  Historical Features"):
    feature_list.append(get_last10_features(row['horse_clean'], row['race_date']))
features = pd.DataFrame(feature_list)
race_results = pd.concat([race_results.reset_index(drop=True), features], axis=1)

# --- Categorical columns for one-hot ---
categorical_cols = [
    'mode_course10', 'mode_jockey10', 'mode_trainer10',
    'mode_race_class10', 'mode_surface10', 'mode_track10', 'mode_going10'
]
race_results = pd.get_dummies(race_results, columns=categorical_cols, dummy_na=True)
race_results = race_results.loc[:, ~race_results.columns.duplicated()]

# --- Feature columns ---
feature_cols = [
    'avg_place10', 'runs10', 'days_since_last', 'avg_odds10', 'avg_distance10'
] + [c for c in race_results.columns if any(col + '_' in c for col in categorical_cols)]
feature_cols = pd.unique(feature_cols)

# --- Remove any columns with all NaN/zero ---
X_train_full = race_results[feature_cols]
X_train = X_train_full.loc[:, X_train_full.var() > 0]  # keep only with variance
used_feature_cols = X_train.columns

train = race_results.dropna(subset=list(used_feature_cols) + ['placed', 'won'])
print(f"Training data prepared: {train.shape[0]} samples.")

X = train[used_feature_cols]
y_place = train['placed']
y_win = train['won']

print("Sample training features:")
print(X.head())
print("Feature variance (should not be all zero):")
print(X.var())

# ---------- Train XGBoost Models ----------
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

# ---------- Feature Engineering for Future Races ----------
print("Feature engineering for future races...")
future_feature_list = []
for idx, row in tqdm(future_races.iterrows(), total=len(future_races), desc="  Future Features"):
    future_feature_list.append(get_last10_features(row['horse_clean'], row['race_date']))
future_features = pd.DataFrame(future_feature_list)
future_races = pd.concat([future_races.reset_index(drop=True), future_features], axis=1)

future_races = pd.get_dummies(future_races, columns=categorical_cols, dummy_na=True)
future_races = future_races.loc[:, ~future_races.columns.duplicated()]

# Align columns with training data
for col in used_feature_cols:
    if col not in future_races.columns:
        future_races[col] = 0
X_future = future_races[used_feature_cols].fillna(-1)

print("Future Features Sample:")
print(X_future.head())

# ---------- Predict Probabilities ----------
future_races['place_probability'] = place_model.predict_proba(X_future)[:,1]
future_races['win_probability'] = win_model.predict_proba(X_future)[:,1]
print("Predictions for future races complete.")

# ---------- Improved Monte Carlo Simulations (ranking) ----------
def monte_carlo_rank_simulation(win_probs, n_sim=10000, top_n=3):
    n = len(win_probs)
    win_probs = np.array(win_probs)
    horses_in_top_n = np.zeros(n)
    wins = np.zeros(n)
    for _ in range(n_sim):
        # Add random Gumbel noise to log odds (random utility model)
        utilities = np.log(win_probs + 1e-10) + np.random.gumbel(size=n)
        ranks = np.argsort(-utilities)  # higher is better
        wins[ranks[0]] += 1
        horses_in_top_n[ranks[:top_n]] += 1
    return wins / n_sim, horses_in_top_n / n_sim

print("Running improved Monte Carlo simulations for future races...")
output = []
race_groups = list(future_races.groupby(['race_date', 'race_no']))
for (race_id, race_df) in tqdm(race_groups, desc="Monte Carlo Races"):
    horses = race_df['horse'].values
    horse_cleans = race_df['horse_clean'].values
    horse_nos = race_df['horse_no'].values
    win_probs = race_df['win_probability'].values

    sim_win_pct, sim_place_pct = monte_carlo_rank_simulation(win_probs, n_sim=10000, top_n=3)

    for horse, horse_clean, h_no, w_prob, sim_w, sim_p in zip(
            horses, horse_cleans, horse_nos, win_probs, sim_win_pct, sim_place_pct
        ):
        output.append({
            'race_date': race_id[0],
            'race_no': race_id[1],
            'horse': horse,
            'horse_clean': horse_clean,
            'horse_no': h_no,
            'win_probability': w_prob,
            'sim_win_pct': sim_w,
            'sim_place_pct': sim_p
        })

sim_results = pd.DataFrame(output)
print("Monte Carlo simulation complete.")

# ---------- Prepare Final Results Table ----------
final_results = sim_results[['race_date', 'race_no', 'horse_no', 'horse', 'win_probability', 'sim_win_pct', 'sim_place_pct']]

# ---------- Save to MySQL ----------
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
