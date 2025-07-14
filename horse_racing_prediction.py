import os
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
from sqlalchemy import create_engine
import json

# Step 1: Check package versions
print("pandas version:", pd.__version__)
#print("xgboost version:", xgboost.__version__)

# Step 2: Connect to MySQL with SQLAlchemy
load_dotenv()
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": "utf8mb4"
}
try:
    engine = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}?charset={db_config['charset']}"
    )
    print("MySQL connection established with SQLAlchemy")
except Exception as e:
    print(f"MySQL connection error: {e}")
    exit(1)

# Step 3: Load historical data
query_historical = """
SELECT 
    id, race_date, course, race_no, horse, jockey, trainer, pla, 
    act_wt, declared_wt, draw_no, lbw, running_position, finish_time, win_odds,
    race_info
FROM race_results
WHERE pla IS NOT NULL 
  AND pla NOT IN ('WD', 'DH', '') 
  AND win_odds IS NOT NULL 
  AND finish_time IS NOT NULL
"""
df = pd.read_sql(query_historical, engine)

# Step 4: Load future data
query_future = """
SELECT 
    race_info, horse_no, horse, draw_no, act_wt, jockey, trainer, 
    win_odds, place_odds, race_date, course, race_no
FROM future_races
"""
new_data = pd.read_sql(query_future, engine)

# Debug: Print dataset sizes and unique values
print(f"Number of rows in race_results: {len(df)}")
print(f"Number of rows in future_races: {len(new_data)}")
print("Unique race_info values in race_results:")
print(df['race_info'].unique())
print("Unique race_info values in future_races:")
print(new_data['race_info'].unique())

# Step 5: Clean jockey and trainer data
df['jockey'] = df['jockey'].replace(['---', ''], np.nan)
df['trainer'] = df['trainer'].replace(['---', ''], np.nan)
new_data['jockey'] = new_data['jockey'].replace(['---', ''], np.nan)
new_data['trainer'] = new_data['trainer'].replace(['---', ''], np.nan)

# Step 6: Parse race_info
def parse_race_info(race_info):
    try:
        race_class = None
        distance = None
        rating_high = None
        rating_low = None
        if not isinstance(race_info, str) or not race_info:
            print(f"Warning: Invalid race_info '{race_info}'")
            return {'race_class': None, 'distance': None, 'rating_high': None, 'rating_low': None}
        
        # Extract distance (e.g., 1200m, 1200, 1.2KM)
        distance_match = re.search(r'(\d+\.?\d*)\s*(?:[Mm]|KM)?', race_info, re.IGNORECASE)
        if distance_match:
            distance = float(distance_match.group(1))
            if distance < 100:
                distance *= 1000
        
        # Extract race class
        if 'griffin race' in race_info.lower():
            race_class = 'Griffin'
        elif 'restricted race' in race_info.lower():
            race_class = 'Restricted'
        elif 'hong kong group one' in race_info.lower():
            race_class = 'Group 1'
        elif 'hong kong group two' in race_info.lower():
            race_class = 'Group 2'
        elif 'hong kong group three' in race_info.lower():
            race_class = 'Group 3'
        elif 'group one' in race_info.lower():
            race_class = 'Group 1'
        elif 'group two' in race_info.lower():
            race_class = 'Group 2'
        elif 'group three' in race_info.lower():
            race_class = 'Group 3'
        elif re.search(r'class \d', race_info.lower()):
            race_class = re.search(r'Class \d', race_info, re.IGNORECASE).group(0)
        
        # Extract rating range (e.g., (60-40), (80+))
        rating_match = re.search(r'\((\d+)-(\d+)\)|\((\d+)\+\)', race_info)
        if rating_match:
            if rating_match.group(3):
                rating_high = int(rating_match.group(3))
                rating_low = rating_high
            else:
                rating_high = int(rating_match.group(1))
                rating_low = int(rating_match.group(2))
        else:
            if race_class:
                if race_class == 'Class 1':
                    rating_high, rating_low = 100, 80
                elif race_class == 'Class 2':
                    rating_high, rating_low = 80, 60
                elif race_class == 'Class 3':
                    rating_high, rating_low = 60, 40
                elif race_class == 'Class 4':
                    rating_high, rating_low = 60, 40
                elif race_class == 'Class 5':
                    rating_high, rating_low = 40, 0
                elif race_class in ['Group 1', 'Group 2', 'Group 3']:
                    rating_high, rating_low = 100, 80
                elif race_class in ['Griffin', 'Restricted']:
                    rating_high, rating_low = 60, 40
        
        return {
            'race_class': race_class,
            'distance': distance,
            'rating_high': rating_high,
            'rating_low': rating_low
        }
    except Exception as e:
        print(f"Error parsing race_info '{race_info}': {str(e)}")
        return {'race_class': None, 'distance': None, 'rating_high': None, 'rating_low': None}

race_info_df = df['race_info'].apply(parse_race_info).apply(pd.Series)
df = pd.concat([df, race_info_df], axis=1)
new_data = pd.concat([new_data, new_data['race_info'].apply(parse_race_info).apply(pd.Series)], axis=1)

# Step 7: Verify critical columns exist
required_columns = ['race_class', 'distance', 'rating_high', 'rating_low']
for col in required_columns:
    if col not in new_data.columns:
        print(f"Warning: '{col}' column not created in future data. Setting to NaN.")
        new_data[col] = None
    if col not in df.columns:
        print(f"Warning: '{col}' column not created in historical data. Setting to NaN.")
        df[col] = None

# Step 8: Convert data types (historical)
df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
df['pla'] = pd.to_numeric(df['pla'], errors='coerce')
df['act_wt'] = pd.to_numeric(df['act_wt'], errors='coerce')
df['declared_wt'] = pd.to_numeric(df['declared_wt'], errors='coerce')
df['draw_no'] = pd.to_numeric(df['draw_no'], errors='coerce')
df['lbw'] = pd.to_numeric(df['lbw'], errors='coerce')
df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
df['rating_high'] = pd.to_numeric(df['rating_high'], errors='coerce')
df['rating_low'] = pd.to_numeric(df['rating_low'], errors='coerce')

def convert_finish_time(time_str):
    try:
        minutes, seconds = time_str.split(':')
        seconds, tenths = seconds.split('.')
        return int(minutes) * 60 + int(seconds) + int(tenths) / 100
    except:
        return None
df['finish_time_secs'] = df['finish_time'].apply(convert_finish_time)

# Step 9: Convert data types (future)
new_data['race_date'] = pd.to_datetime(new_data['race_date'], errors='coerce')
new_data['act_wt'] = pd.to_numeric(new_data['act_wt'], errors='coerce')
new_data['draw_no'] = pd.to_numeric(new_data['draw_no'], errors='coerce')
new_data['win_odds'] = pd.to_numeric(new_data['win_odds'], errors='coerce')
new_data['place_odds'] = pd.to_numeric(new_data['place_odds'], errors='coerce')
new_data['distance'] = pd.to_numeric(new_data['distance'], errors='coerce')
new_data['rating_high'] = pd.to_numeric(new_data['rating_high'], errors='coerce')
new_data['rating_low'] = pd.to_numeric(new_data['rating_low'], errors='coerce')

# Step 10: Create target variables (historical)
df['is_winner'] = (df['pla'] == 1).astype(int)
df['is_placed'] = df['pla'].isin([1, 2, 3]).astype(int)

# Debug: Check target variable distribution
print("is_winner distribution:", df['is_winner'].value_counts().to_dict())
print("is_placed distribution:", df['is_placed'].value_counts().to_dict())

# Step 11: Feature engineering (historical) - Part 1
# Compute speed and win/place rates before dropna
df['speed'] = df['distance'] / df['finish_time_secs']
df_historical = df.copy()

for col in ['horse', 'jockey', 'trainer']:
    df[f'{col}_win_rate'] = df.groupby(col)['is_winner'].transform('mean')
    df[f'{col}_place_rate'] = df.groupby(col)['is_placed'].transform('mean')

# Calculate jockey_trainer rates
jockey_trainer_counts = df[df['jockey'].notna() & df['trainer'].notna()].groupby(['jockey', 'trainer']).size().reset_index(name='race_count')
jockey_trainer_win = df[df['jockey'].notna() & df['trainer'].notna()].groupby(['jockey', 'trainer'])['is_winner'].mean().reset_index(name='jockey_trainer_win_rate')
jockey_trainer_win = jockey_trainer_win.merge(jockey_trainer_counts, on=['jockey', 'trainer'])
jockey_trainer_place = df[df['jockey'].notna() & df['trainer'].notna()].groupby(['jockey', 'trainer'])['is_placed'].mean().reset_index(name='jockey_trainer_place_rate')
jockey_trainer_win = jockey_trainer_win.merge(jockey_trainer_place[['jockey', 'trainer', 'jockey_trainer_place_rate']], on=['jockey', 'trainer'])
# Improve jockey_trainer_win_rate for race_count=1
jockey_trainer_win['jockey_trainer_win_rate'] = jockey_trainer_win.apply(
    lambda row: df[df['jockey'] == row['jockey']]['jockey_win_rate'].mean() if row['race_count'] == 1 else row['jockey_trainer_win_rate'], axis=1)

# Debug: Check win rate distributions before dropna
print("Unique horse_win_rate values:", df['horse_win_rate'].nunique(), "Sample:", df['horse_win_rate'].head(10).tolist())
print("Unique jockey_win_rate values:", df['jockey_win_rate'].nunique(), "Sample:", df['jockey_win_rate'].head(10).tolist())
print("Unique trainer_win_rate values:", df['trainer_win_rate'].nunique(), "Sample:", df['trainer_win_rate'].head(10).tolist())
print(f"jockey_trainer_win DataFrame shape: {jockey_trainer_win.shape}")
print("Sample jockey_trainer_win rows:", jockey_trainer_win.head().to_dict())
print("Jockey-trainer pair frequency:", jockey_trainer_counts.head(10).to_dict())

# Step 12: Drop rows with missing critical values (historical)
df = df.dropna(subset=['pla', 'win_odds', 'finish_time_secs', 'distance', 'race_class'])
print(f"Number of rows in df after dropna: {len(df)}")
print("Post-dropna is_winner distribution:", df['is_winner'].value_counts().to_dict())
print("Post-dropna is_placed distribution:", df['is_placed'].value_counts().to_dict())
print(f"Missing jockey values: {df['jockey'].isna().sum()}")
print(f"Missing trainer values: {df['trainer'].isna().sum()}")

# Step 13: Feature engineering (historical) - Part 2
df = df.sort_values(['horse', 'race_date'])
df['avg_pla_last_5'] = df.groupby('horse')['pla'].shift(1).rolling(5, min_periods=1).mean()
df['place_rate_last_5'] = df.groupby('horse')['is_placed'].shift(1).rolling(5, min_periods=1).mean()
df['avg_speed_last_5'] = df.groupby('horse')['speed'].shift(1).rolling(5, min_periods=1).mean()
df['avg_lbw_last_5'] = df.groupby('horse')['lbw'].shift(1).rolling(5, min_periods=1).mean()
df['races_last_30_days'] = df.groupby('horse')['race_date'].transform(
    lambda x: ((x - x.shift(1)).dt.days < 30).rolling(5, min_periods=1).sum()
)

# Merge jockey_trainer and course features
df = df.merge(jockey_trainer_win[['jockey', 'trainer', 'jockey_trainer_win_rate', 'jockey_trainer_place_rate']], on=['jockey', 'trainer'], how='left')
# Impute missing jockey_trainer rates with jockey mean, then overall mean
df['jockey_trainer_win_rate'] = df['jockey_trainer_win_rate'].fillna(df.groupby('jockey')['jockey_trainer_win_rate'].transform('mean'))
df['jockey_trainer_win_rate'] = df['jockey_trainer_win_rate'].fillna(df['jockey_trainer_win_rate'].mean())
df['jockey_trainer_place_rate'] = df['jockey_trainer_place_rate'].fillna(df.groupby('jockey')['jockey_trainer_place_rate'].transform('mean'))
df['jockey_trainer_place_rate'] = df['jockey_trainer_place_rate'].fillna(df['jockey_trainer_place_rate'].mean())
avg_pla_course = df.groupby(['horse', 'course'])['pla'].mean().reset_index(name='avg_pla_course')
place_rate_course = df.groupby(['horse', 'course'])['is_placed'].mean().reset_index(name='place_rate_course')
df = df.merge(avg_pla_course, on=['horse', 'course'], how='left')
df = df.merge(place_rate_course, on=['horse', 'course'], how='left')

# Debug: Check jockey_trainer merges
print("Columns after jockey_trainer merges:", df.columns.tolist())
print("Sample jockey_trainer_win_rate values:", df['jockey_trainer_win_rate'].head(10).tolist())
print("Sample jockey_trainer_place_rate values:", df['jockey_trainer_place_rate'].head(10).tolist())
print("jockey_trainer_win_rate distribution:", df['jockey_trainer_win_rate'].value_counts().head(10).to_dict())
print("jockey_trainer_win_rate summary: min=%.4f, max=%.4f, mean=%.4f, std=%.4f" % (
    df['jockey_trainer_win_rate'].min(), df['jockey_trainer_win_rate'].max(),
    df['jockey_trainer_win_rate'].mean(), df['jockey_trainer_win_rate'].std()))
print("Unmatched jockey_trainer_win merges:", df['jockey_trainer_win_rate'].isna().sum())
print("Unmatched jockey-trainer pairs:", df[df['jockey_trainer_win_rate'].isna()][['jockey', 'trainer']].drop_duplicates().to_dict())

# Drop horse, jockey, trainer columns
df = df.drop(['horse', 'jockey', 'trainer'], axis=1)

# Debug: Print columns after dropping
print("Columns in df after dropping horse, jockey, trainer:", df.columns.tolist())

# Create course_draw feature
df['course_draw'] = df['course'] + '_' + df['draw_no'].astype(str)

# Debug: Print columns after creating course_draw
print("Columns in df after creating course_draw:", df.columns.tolist())

# Convert course to dummy variables and continue with remaining features
df = pd.get_dummies(df, columns=['course'], prefix='course')
df = pd.get_dummies(df, columns=['course_draw'], prefix='course_draw')
race_class_map = {
    'Class 1': 1, 'Class 2': 2, 'Class 3': 3, 'Class 4': 4, 'Class 5': 5,
    'Group 1': 0, 'Group 2': 1, 'Group 3': 2, 'Griffin': 6, 'Restricted': 6
}
df['race_class'] = df['race_class'].map(race_class_map)
df['year'] = df['race_date'].dt.year
df['month'] = df['race_date'].dt.month
df['day_of_week'] = df['race_date'].dt.dayofweek
def parse_running_position(rp):
    try:
        if isinstance(rp, str) and rp.strip():
            positions = rp.strip().split()
            if positions and positions[0].isdigit():
                return int(positions[0])
            else:
                print(f"Warning: Invalid running_position '{rp}'")
        return None
    except Exception as e:
        print(f"Error parsing running_position '{rp}': {str(e)}")
        return None
df['start_position'] = df['running_position'].apply(parse_running_position)
df['start_position'] = df['start_position'].fillna(df['start_position'].mean() if df['start_position'].notna().any() else 1)
df['odds_relative_win'] = df.groupby(['race_date', 'race_no'])['win_odds'].transform(lambda x: x / x.mean())

# Debug: Print unique values for problematic features
print("Unique start_position values:", df['start_position'].unique())
print(f"Number of unique jockeys: {df_historical['jockey'].nunique()}")
print(f"Number of unique trainers: {df_historical['trainer'].nunique()}")

# Step 14: Handle missing values (historical)
for col in ['avg_pla_last_5', 'place_rate_last_5', 'avg_speed_last_5', 'jockey_win_rate', 
            'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate', 
            'jockey_trainer_win_rate', 'jockey_trainer_place_rate', 'avg_pla_course', 
            'place_rate_course', 'avg_lbw_last_5', 'rating_high', 'rating_low']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean() if df[col].notna().any() else 0)

# Step 15: Standardize features (historical)
numerical_features = ['win_odds', 'act_wt', 'declared_wt', 'draw_no', 'distance', 'race_class', 
                      'rating_high', 'rating_low', 'start_position', 'avg_pla_last_5', 
                      'place_rate_last_5', 'avg_speed_last_5', 'jockey_win_rate', 
                      'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate', 
                      'jockey_trainer_win_rate', 'jockey_trainer_place_rate', 'avg_pla_course', 
                      'place_rate_course', 'odds_relative_win', 'races_last_30_days', 'avg_lbw_last_5']

# Debug: Check for zero variance or invalid values
print("Checking numerical features for issues:")
for col in numerical_features:
    if col in df.columns:
        if df[col].nunique() <= 1:
            print(f"Warning: Feature '{col}' has zero or near-zero variance (unique values: {df[col].nunique()})")
        if df[col].isna().any() or np.isinf(df[col]).any():
            print(f"Warning: Feature '{col}' contains NaN or infinite values")
    else:
        print(f"Warning: Feature '{col}' not found in df")

# Remove features with zero variance or replace NaN/infinite values
valid_numerical_features = [col for col in numerical_features if col in df.columns and df[col].nunique() > 1]
df[valid_numerical_features] = df[valid_numerical_features].replace([np.inf, -np.inf], np.nan).fillna(df[valid_numerical_features].mean())

# Debug: Print valid_numerical_features
print("valid_numerical_features:", valid_numerical_features)

scaler = StandardScaler()
df[valid_numerical_features] = scaler.fit_transform(df[valid_numerical_features])

# Step 16: Feature engineering (future)
new_data = new_data.merge(
    df_historical.groupby('horse')[['pla', 'is_placed', 'speed']].mean().reset_index().rename(
        columns={'pla': 'avg_pla_last_5', 'is_placed': 'place_rate_last_5', 'speed': 'avg_speed_last_5'}),
    on='horse', how='left')
new_data = new_data.merge(
    df_historical[df_historical['jockey'].notna()].groupby('jockey')[['is_winner', 'is_placed']].mean().reset_index().rename(
        columns={'is_winner': 'jockey_win_rate', 'is_placed': 'jockey_place_rate'}),
    on='jockey', how='left')
new_data = new_data.merge(
    df_historical[df_historical['trainer'].notna()].groupby('trainer')[['is_winner', 'is_placed']].mean().reset_index().rename(
        columns={'is_winner': 'trainer_win_rate', 'is_placed': 'trainer_place_rate'}),
    on='trainer', how='left')
new_data = new_data.merge(
    df_historical[df_historical['jockey'].notna() & df_historical['trainer'].notna()].groupby(['jockey', 'trainer'])['is_winner'].mean().reset_index(name='jockey_trainer_win_rate'),
    on=['jockey', 'trainer'], how='left')
new_data = new_data.merge(
    df_historical[df_historical['jockey'].notna() & df_historical['trainer'].notna()].groupby(['jockey', 'trainer'])['is_placed'].mean().reset_index(name='jockey_trainer_place_rate'),
    on=['jockey', 'trainer'], how='left')
new_data = new_data.merge(
    df_historical.groupby(['horse', 'course'])['pla'].mean().reset_index(name='avg_pla_course'),
    on=['horse', 'course'], how='left')
new_data = new_data.merge(
    df_historical.groupby(['horse', 'course'])['is_placed'].mean().reset_index(name='place_rate_course'),
    on=['horse', 'course'], how='left')
new_data = new_data.merge(
    df_historical.groupby('horse')['lbw'].mean().reset_index(name='avg_lbw_last_5'), on='horse', how='left')

# Impute missing jockey_trainer rates in new_data
new_data['jockey_trainer_win_rate'] = new_data['jockey_trainer_win_rate'].fillna(new_data.groupby('jockey')['jockey_trainer_win_rate'].transform('mean'))
new_data['jockey_trainer_win_rate'] = new_data['jockey_trainer_win_rate'].fillna(df['jockey_trainer_win_rate'].mean())
new_data['jockey_trainer_place_rate'] = new_data['jockey_trainer_place_rate'].fillna(new_data.groupby('jockey')['jockey_trainer_place_rate'].transform('mean'))
new_data['jockey_trainer_place_rate'] = new_data['jockey_trainer_place_rate'].fillna(df['jockey_trainer_place_rate'].mean())

# Create course_draw and dummy variables
new_data['course_draw'] = new_data['course'] + '_' + new_data['draw_no'].astype(str)
new_data = pd.get_dummies(new_data, columns=['course'], prefix='course')
new_data = pd.get_dummies(new_data, columns=['course_draw'], prefix='course_draw')
new_data['race_class'] = new_data['race_class'].map(race_class_map)
new_data['odds_relative_win'] = new_data.groupby(['race_date', 'race_no'])['win_odds'].transform(lambda x: x / x.mean())
new_data['odds_relative_place'] = new_data.groupby(['race_date', 'race_no'])['place_odds'].transform(lambda x: x / x.mean())
new_data['races_last_30_days'] = new_data.merge(
    df_historical.groupby('horse')['race_date'].apply(
        lambda x: ((pd.to_datetime('2025-07-15') - x).dt.days < 30).sum()
    ).reset_index(name='races_last_30_days'), on='horse', how='left')['races_last_30_days']

# Step 17: Handle missing values (future)
for col in ['avg_pla_last_5', 'place_rate_last_5', 'avg_speed_last_5', 'jockey_win_rate', 
            'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate', 
            'jockey_trainer_win_rate', 'jockey_trainer_place_rate', 'avg_pla_course', 
            'place_rate_course', 'avg_lbw_last_5', 'rating_high', 'rating_low', 'distance']:
    new_data[col] = new_data[col].fillna(df[col].mean() if col in df.columns else 0)

# Step 18: Standardize future data
new_data_numerical_features = valid_numerical_features.copy()
for col in new_data_numerical_features:
    if col not in new_data.columns:
        print(f"Adding missing column '{col}' to new_data with mean value from df")
        new_data[col] = df[col].mean() if col in df.columns else 0

# Debug: Print feature lists and model input
print("new_data_numerical_features:", new_data_numerical_features)
print("Columns in new_data before standardization:", new_data.columns.tolist())
print("X_train_win columns:", valid_numerical_features + [col for col in df.columns if col.startswith('course_') or col.startswith('course_draw_')])
print("X_train_win dtypes:", df[valid_numerical_features + [col for col in df.columns if col.startswith('course_') or col.startswith('course_draw_')]].dtypes)

new_data[new_data_numerical_features] = new_data[new_data_numerical_features].replace([np.inf, -np.inf], np.nan).fillna(df[new_data_numerical_features].mean())
new_data[new_data_numerical_features] = scaler.transform(new_data[new_data_numerical_features])

# Step 19: Align columns
missing_cols = set(df.columns) - set(new_data.columns)
for col in missing_cols:
    if col.startswith('course_') or col.startswith('course_draw_'):
        new_data[col] = 0

# Step 20: Train models
features = valid_numerical_features + [col for col in df.columns if col.startswith('course_') or col.startswith('course_draw_')]
X_win = df[features]
y_win = df['is_winner']
X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(X_win, y_win, test_size=0.2, random_state=42)
model_win = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_win.fit(X_train_win, y_train_win)
y_pred_win = model_win.predict_proba(X_test_win)[:, 1]
print(f"Win ROC AUC: {roc_auc_score(y_test_win, y_pred_win):.3f}")

X_place = df[features]
y_place = df['is_placed']
X_train_place, X_test_place, y_train_place, y_test_place = train_test_split(X_place, y_place, test_size=0.2, random_state=42)
model_place = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_place.fit(X_train_place, y_train_place)
y_pred_place = model_place.predict_proba(X_test_place)[:, 1]
print(f"Place ROC AUC: {roc_auc_score(y_test_place, y_pred_place):.3f}")

# Step 21: Predict on future data
new_data['win_probability'] = model_win.predict_proba(new_data[features])[:, 1]
new_data['place_probability'] = model_place.predict_proba(new_data[features])[:, 1]

# Step 22: Save predictions to CSV
new_data[['race_date', 'race_no', 'horse_no', 'horse', 'win_probability', 'place_probability']].to_csv(
    'predictions.csv', index=False
)
print("Predictions saved to predictions.csv")

# Step 23: Save predictions to MySQL
try:
    with engine.connect() as conn:
        new_data[['race_date', 'race_no', 'horse_no', 'horse', 'win_probability', 'place_probability']].to_sql(
            'race_predictions', conn, if_exists='replace', index=False
        )
    print("Predictions saved to MySQL table race_predictions")
except Exception as e:
    print(f"Error saving to MySQL: {e}")

# Step 24: Visualize jockey_trainer_win_rate distribution
hist, bins = np.histogram(df['jockey_trainer_win_rate'], bins=30)
hist_data = {
    "bins": bins.tolist(),
    "counts": hist.tolist(),
    "bin_edges": [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
}
print("jockey_trainer_win_rate histogram (30 bins):", json.dumps(hist_data, indent=2))