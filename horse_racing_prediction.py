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
    # Remove everything in parentheses and trailing whitespace
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
        SELECT MAX(id) - 199 FROM racedates
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
print("Feature engineering for historical races with progress bar (manual loop)...")
def get_last_features(horse_clean, date):
    prev = race_results[(race_results['horse_clean'] == horse_clean) & (race_results['race_date'] < date)]
    prev = prev.sort_values('race_date', ascending=False)
    if prev.empty:
        return pd.Series({'last_place': np.nan, 'last_odds': np.nan, 'avg_place3': np.nan, 'runs': 0, 'days_since_last': np.nan,
                          'race_class': None, 'distance': None, 'surface': None, 'track': None, 'going': None})
    last = prev.iloc[0]
    avg_place3 = prev.head(3)['pla_num'].mean()
    days_since_last = (pd.to_datetime(date) - pd.to_datetime(last['race_date'])).days
    return pd.Series({
        'last_place': last['pla_num'],
        'last_odds': pd.to_numeric(last['win_odds'], errors='coerce'),
        'avg_place3': avg_place3,
        'runs': len(prev),
        'days_since_last': days_since_last,
        'race_class': last.get('race_class', None),
        'distance': last.get('distance', None),
        'surface': last.get('surface', None),
        'track': last.get('track', None),
        'going': last.get('going', None)
    })

feature_list = []
for idx, row in tqdm(race_results.iterrows(), total=len(race_results), desc="  Historical Features"):
    feature_list.append(get_last_features(row['horse_clean'], row['race_date']))
features = pd.DataFrame(feature_list)
race_results = pd.concat([race_results.reset_index(drop=True), features], axis=1)

# One-hot encode new fields for model
categorical_cols = ['race_class', 'surface', 'track', 'going']
race_results = pd.get_dummies(race_results, columns=categorical_cols, dummy_na=True)
race_results = race_results.loc[:, ~race_results.columns.duplicated()]
feature_cols = ['last_place', 'last_odds', 'avg_place3', 'runs', 'days_since_last', 'distance'] \
               + [c for c in race_results.columns if any(col + '_' in c for col in categorical_cols)]
feature_cols = pd.unique(feature_cols)

train = race_results.dropna(subset=['last_place', 'last_odds', 'avg_place3', 'placed', 'won', 'distance'])
print(f"Training data prepared: {train.shape[0]} samples.")

X = train[feature_cols]
y_place = train['placed']
y_win = train['won']

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

# ---------- Feature Engineering for Future Races (using horse_clean) ----------
print("Feature engineering for future races...")
future_feature_list = []
for idx, row in tqdm(future_races.iterrows(), total=len(future_races), desc="  Future Features"):
    future_feature_list.append(get_last_features(row['horse_clean'], row['race_date']))
future_features = pd.DataFrame(future_feature_list)
future_races = pd.concat([future_races.reset_index(drop=True), future_features], axis=1)

# One-hot encode for future races, matching historical data columns
future_races = pd.get_dummies(future_races, columns=categorical_cols, dummy_na=True)
future_races = future_races.loc[:, ~future_races.columns.duplicated()]

# Align columns with training data (add missing columns with zeros, but don't overwrite DataFrame!)
for col in feature_cols:
    if col not in future_races.columns:
        future_races[col] = 0
X_future = future_races[feature_cols].fillna(-1)

# ！！！metadata 唔 overwrite，保持所有 column
future_races['place_probability'] = place_model.predict_proba(X_future)[:,1]
future_races['win_probability'] = win_model.predict_proba(X_future)[:,1]
print("Predictions for future races complete.")

# ---------- Monte Carlo Simulations ----------
def monte_carlo_place_simulation(place_probs, n_sim=10000):
    n = len(place_probs)
    results = np.zeros(n)
    for _ in range(n_sim):
        placed = np.random.rand(n) < place_probs
        results += placed
    return results / n_sim

def monte_carlo_win_simulation(win_probs, n_sim=10000):
    n = len(win_probs)
    results = np.zeros(n)
    win_probs = np.array(win_probs)
    if win_probs.sum() == 0:
        return results
    for _ in range(n_sim):
        winner = np.random.choice(n, p=win_probs / win_probs.sum())
        results[winner] += 1
    return results / n_sim

print("Running Monte Carlo simulations for future races...")
output = []
# 🟢 Now groupby is SAFE!
race_groups = list(future_races.groupby(['race_date', 'race_no']))
for (race_id, race_df) in tqdm(race_groups, desc="Monte Carlo Races"):
    horses = race_df['horse'].values
    horse_cleans = race_df['horse_clean'].values
    horse_nos = race_df['horse_no'].values
    place_probs = race_df['place_probability'].values
    win_probs = race_df['win_probability'].values

    sim_place_pct = monte_carlo_place_simulation(place_probs, n_sim=10000)
    sim_win_pct = monte_carlo_win_simulation(win_probs, n_sim=10000)

    for horse, horse_clean, h_no, p_prob, w_prob, sim_p, sim_w in zip(
            horses, horse_cleans, horse_nos, place_probs, win_probs, sim_place_pct, sim_win_pct
        ):
        output.append({
            'race_date': race_id[0],
            'race_no': race_id[1],
            'horse': horse,
            'horse_clean': horse_clean,
            'horse_no': h_no,
            'win_probability': w_prob,
            'place_probability': p_prob,
            'sim_win_pct': sim_w,
            'sim_place_pct': sim_p
        })

sim_results = pd.DataFrame(output)
print("Monte Carlo simulation complete.")

# ---------- Prepare Final Results Table ----------
final_results = sim_results[['race_date', 'race_no', 'horse_no', 'horse', 'win_probability', 'place_probability', 'sim_win_pct', 'sim_place_pct']]

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
