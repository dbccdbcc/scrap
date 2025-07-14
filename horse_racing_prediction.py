# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 01:03:04 2025

@author: User
"""

import pandas as pd
import os
from dotenv import load_dotenv
import pymysql
import re
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Step 1: Connect to MySQL
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": "utf8mb4"
}
# Setup MySQL connection
conn = pymysql.connect(**db_config)
cursor = conn.cursor()


# Step 2: Load historical data
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
df = pd.read_sql(query_historical, conn)

# Step 3: Load future data
query_future = """
SELECT 
    race_info, horse_no, horse, draw_no, act_wt, jockey, trainer, 
    win_odds, place_odds, race_date, course, race_no
FROM future_races
"""
new_data = pd.read_sql(query_future, conn)
conn.close()

# Step 4: Parse race_info
def parse_race_info(race_info):
    try:
        race_class = None
        distance = None
        rating_high = None
        rating_low = None
        distance_match = re.search(r'(\d+)M', race_info)
        if distance_match:
            distance = int(distance_match.group(1))
        if 'Griffin Race' in race_info:
            race_class = 'Griffin'
        elif 'Restricted Race' in race_info:
            race_class = 'Restricted'
        elif 'Hong Kong Group One' in race_info:
            race_class = 'Group 1'
        elif 'Hong Kong Group Two' in race_info:
            race_class = 'Group 2'
        elif 'Hong Kong Group Three' in race_info:
            race_class = 'Group 3'
        elif 'Group One' in race_info:
            race_class = 'Group 1'
        elif 'Group Two' in race_info:
            race_class = 'Group 2'
        elif 'Group Three' in race_info:
            race_class = 'Group 3'
        elif re.match(r'Class \d', race_info):
            race_class = re.search(r'Class \d', race_info).group(0)
        rating_match = re.search(r'\((\d+)-(\d+)\)|\((\d+)\+\)', race_info)
        if rating_match:
            if rating_match.group(3):
                rating_high = int(rating_match.group(3))
                rating_low = rating_high
            else:
                rating_high = int(rating_match.group(1))
                rating_low = int(rating_match.group(2))
        return {
            'race_class': race_class,
            'distance': distance,
            'rating_high': rating_high,
            'rating_low': rating_low
        }
    except:
        return {'race_class': None, 'distance': None, 'rating_high': None, 'rating_low': None}

# Apply to historical data
race_info_df = df['race_info'].apply(parse_race_info).apply(pd.Series)
df = pd.concat([df, race_info_df], axis=1)

# Apply to future data
new_data = pd.concat([new_data, new_data['race_info'].apply(parse_race_info).apply(pd.Series)], axis=1)

# Step 5: Convert data types (historical)
df['race_date'] = pd.to_datetime(df['race_date'])
df['pla'] = pd.to_numeric(df['pla'], errors='coerce')
df['act_wt'] = pd.to_numeric(df['act_wt'], errors='coerce')
df['declared_wt'] = pd.to_numeric(df['declared_wt'], errors='coerce')
df['draw_no'] = pd.to_numeric(df['draw_no'], errors='coerce')
df['lbw'] = pd.to_numeric(df['lbw'], errors='coerce')
df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
df['rating_high'] = pd.to_numeric(df['rating_high'], errors='coerce')
df['rating_low'] = pd.to_numeric(df['rating_low'], errors='coerce')

# Convert finish_time to seconds
def convert_finish_time(time_str):
    try:
        minutes, seconds = time_str.split(':')
        seconds, tenths = seconds.split('.')
        return int(minutes) * 60 + int(seconds) + int(tenths) / 100
    except:
        return None
df['finish_time_secs'] = df['finish_time'].apply(convert_finish_time)

# Step 6: Convert data types (future)
new_data['race_date'] = pd.to_datetime(new_data['race_date'])
new_data['act_wt'] = pd.to_numeric(new_data['act_wt'], errors='coerce')
new_data['draw_no'] = pd.to_numeric(new_data['draw_no'], errors='coerce')
new_data['win_odds'] = pd.to_numeric(new_data['win_odds'], errors='coerce')
new_data['place_odds'] = pd.to_numeric(new_data['place_odds'], errors='coerce')
new_data['distance'] = pd.to_numeric(new_data['distance'], errors='coerce')
new_data['rating_high'] = pd.to_numeric(new_data['rating_high'], errors='coerce')
new_data['rating_low'] = pd.to_numeric(new_data['rating_low'], errors='coerce')

# Step 7: Create target variables (historical)
df['is_winner'] = (df['pla'] == 1).astype(int)
df['is_placed'] = df['pla'].isin([1, 2, 3]).astype(int)

# Step 8: Drop rows with missing critical values (historical)
df = df.dropna(subset=['pla', 'act_wt', 'declared_wt', 'draw_no', 'lbw', 'win_odds', 
                      'finish_time_secs', 'distance', 'race_class'])

# Step 9: Feature engineering (historical)
# Raw features
df = pd.get_dummies(df, columns=['course'], prefix='course')
race_class_map = {
    'Class 1': 1, 'Class 2': 2, 'Class 3': 3, 'Class 4': 4, 'Class 5': 5,
    'Group 1': 0, 'Group 2': 1, 'Group 3': 2, 'Griffin': 6, 'Restricted': 6
}
df['race_class'] = df['race_class'].map(race_class_map)
for col in ['horse', 'jockey', 'trainer']:
    df[f'{col}_win_rate'] = df.groupby(col)['is_winner'].transform('mean')
    df[f'{col}_place_rate'] = df.groupby(col)['is_placed'].transform('mean')
df = df.drop(['horse', 'jockey', 'trainer'], axis=1)
df['year'] = df['race_date'].dt.year
df['month'] = df['race_date'].dt.month
df['day_of_week'] = df['race_date'].dt.dayofweek
df['start_position'] = df['running_position'].apply(lambda x: int(x.split('-')[0]) if x and '-' in x else None)
df['start_position'] = df['start_position'].fillna(df['start_position'].mean())

# Derived features
df = df.sort_values(['horse', 'race_date'])
df['avg_pla_last_5'] = df.groupby('horse')['pla'].shift(1).rolling(5, min_periods=1).mean()
df['place_rate_last_5'] = df.groupby('horse')['is_placed'].shift(1).rolling(5, min_periods=1).mean()
df['speed'] = df['distance'] / df['finish_time_secs']
df['avg_speed_last_5'] = df.groupby('horse')['speed'].shift(1).rolling(5, min_periods=1).mean()
jockey_trainer_win = df.groupby(['jockey', 'trainer'])['is_winner'].mean().reset_index(name='jockey_trainer_win_rate')
jockey_trainer_place = df.groupby(['jockey', 'trainer'])['is_placed'].mean().reset_index(name='jockey_trainer_place_rate')
df = df.merge(jockey_trainer_win, on=['jockey', 'trainer'], how='left')
df = df.merge(jockey_trainer_place, on=['jockey', 'trainer'], how='left')
avg_pla_course = df.groupby(['horse', 'course'])['pla'].mean().reset_index(name='avg_pla_course')
place_rate_course = df.groupby(['horse', 'course'])['is_placed'].mean().reset_index(name='place_rate_course')
df = df.merge(avg_pla_course, on=['horse', 'course'], how='left')
df = df.merge(place_rate_course, on=['horse', 'course'], how='left')
df['odds_relative_win'] = df.groupby(['race_date', 'race_no'])['win_odds'].transform(lambda x: x / x.mean())
df['races_last_30_days'] = df.groupby('horse')['race_date'].transform(
    lambda x: ((x - x.shift(1)).dt.days < 30).rolling(5, min_periods=1).sum()
)
df['course_draw'] = df['course'] + '_' + df['draw_no'].astype(str)
df = pd.get_dummies(df, columns=['course_draw'], prefix='course_draw')
df['avg_lbw_last_5'] = df.groupby('horse')['lbw'].shift(1).rolling(5, min_periods=1).mean()

# Step 10: Handle missing values (historical)
for col in ['avg_pla_last_5', 'place_rate_last_5', 'avg_speed_last_5', 'jockey_win_rate', 
            'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate', 
            'jockey_trainer_win_rate', 'jockey_trainer_place_rate', 'avg_pla_course', 
            'place_rate_course', 'avg_lbw_last_5', 'rating_high', 'rating_low']:
    df[col] = df[col].fillna(df[col].mean())

# Step 11: Standardize features (historical)
scaler = StandardScaler()
numerical_features = ['win_odds', 'act_wt', 'declared_wt', 'draw_no', 'distance', 'race_class', 
                      'rating_high', 'rating_low', 'start_position', 'avg_pla_last_5', 
                      'place_rate_last_5', 'avg_speed_last_5', 'jockey_win_rate', 
                      'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate', 
                      'jockey_trainer_win_rate', 'jockey_trainer_place_rate', 'avg_pla_course', 
                      'place_rate_course', 'odds_relative_win', 'races_last_30_days', 'avg_lbw_last_5']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Step 12: Feature engineering (future)
new_data = pd.get_dummies(new_data, columns=['course'], prefix='course')
new_data['race_class'] = new_data['race_class'].map(race_class_map)
new_data = new_data.merge(
    df.groupby('horse')[['pla', 'is_placed', 'speed']].mean().reset_index().rename(
        columns={'pla': 'avg_pla_last_5', 'is_placed': 'place_rate_last_5', 'speed': 'avg_speed_last_5'}),
    on='horse', how='left')
new_data = new_data.merge(
    df.groupby('jockey')[['is_winner', 'is_placed']].mean().reset_index().rename(
        columns={'is_winner': 'jockey_win_rate', 'is_placed': 'jockey_place_rate'}),
    on='jockey', how='left')
new_data = new_data.merge(
    df.groupby('trainer')[['is_winner', 'is_placed']].mean().reset_index().rename(
        columns={'is_winner': 'trainer_win_rate', 'is_placed': 'trainer_place_rate'}),
    on='trainer', how='left')
new_data = new_data.merge(
    df.groupby(['jockey', 'trainer'])[['is_winner', 'is_placed']].mean().reset_index().rename(
        columns={'is_winner': 'jockey_trainer_win_rate', 'is_placed': 'jockey_trainer_place_rate'}),
    on=['jockey', 'trainer'], how='left')
new_data = new_data.merge(
    df.groupby(['horse', 'course'])[['pla', 'is_placed']].mean().reset_index().rename(
        columns={'pla': 'avg_pla_course', 'is_placed': 'place_rate_course'}),
    on=['horse', 'course'], how='left')
new_data = new_data.merge(
    df.groupby('horse')['lbw'].mean().reset_index(name='avg_lbw_last_5'), on='horse', how='left')
new_data['odds_relative_win'] = new_data.groupby(['race_date', 'race_no'])['win_odds'].transform(lambda x: x / x.mean())
new_data['odds_relative_place'] = new_data.groupby(['race_date', 'race_no'])['place_odds'].transform(lambda x: x / x.mean())
new_data['races_last_30_days'] = new_data.merge(
    df.groupby('horse')['race_date'].apply(
        lambda x: ((pd.to_datetime('2025-07-15') - x).dt.days < 30).sum()
    ).reset_index(name='races_last_30_days'), on='horse', how='left')['races_last_30_days']
new_data['course_draw'] = new_data['course'] + '_' + new_data['draw_no'].astype(str)
new_data = pd.get_dummies(new_data, columns=['course_draw'], prefix='course_draw')

# Step 13: Handle missing values (future)
for col in ['avg_pla_last_5', 'place_rate_last_5', 'avg_speed_last_5', 'jockey_win_rate', 
            'jockey_place_rate', 'trainer_win_rate', 'trainer_place_rate', 
            'jockey_trainer_win_rate', 'jockey_trainer_place_rate', 'avg_pla_course', 
            'place_rate_course', 'avg_lbw_last_5', 'rating_high', 'rating_low']:
    new_data[col] = new_data[col].fillna(df[col].mean() if col in df.columns else 0)

# Step 14: Standardize future data
numerical_features.extend(['place_odds', 'odds_relative_place'])
new_data[numerical_features] = scaler.transform(new_data[numerical_features])

# Step 15: Train models
features = numerical_features + [col for col in df.columns if col.startswith('course_') or col.startswith('course_draw_')]
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

# Step 16: Predict on future data
new_data['win_probability'] = model_win.predict_proba(new_data[features])[:, 1]
new_data['place_probability'] = model_place.predict_proba(new_data[features])[:, 1]

# Step 17: Save predictions to CSV
new_data[['race_date', 'race_no', 'horse_no', 'horse', 'win_probability', 'place_probability']].to_csv(
    'predictions.csv', index=False
)
print("Predictions saved to predictions.csv")

# Step 18: Optionally save predictions back to MySQL
with mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="your_database"
) as conn:
    new_data[['race_date', 'race_no', 'horse_no', 'horse', 'win_probability', 'place_probability']].to_sql(
        'race_predictions', conn, if_exists='replace', index=False
    )
print("Predictions saved to MySQL table race_predictions")