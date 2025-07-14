🏇 HKJC Race Results Scraper
This project scrapes detailed Hong Kong Jockey Club (HKJC) local race result data and inserts it into a MySQL database. It supports efficient scraping by date ID, racecourse (ST/HV), full horse result tables, and avoids duplication using raceDateId foreign key logic.

📦 Features
Scrapes race result data from:
https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx
Supports both Sha Tin (ST) and Happy Valley (HV) racecourses

Extracts full race tables including:

Placing, Horse No, Name, Jockey, Trainer, Weights, Draw, Odds, etc.

Automatically skips unavailable race pages

Inserts data directly into a MySQL database with referential integrity

Uses .env for secure DB credential handling

Prevents reprocessing via raceDateId safeguard

Configurable scraping range via START_ID / END_ID

📁 Folder Structure

project/
├── scraper.py           # Main scraper script
├── .env                 # Environment variables (NOT committed)
├── requirements.txt     # Python dependencies
└── README.md            # You're here
🔧 Requirements
Python 3.8+

Google Chrome + chromedriver

MySQL 5.7/8.0

Required Python packages:

pip install -r requirements.txt
✅ Environment Setup

Create a .env file:

DB_HOST=localhost
DB_USER=your_mysql_user
DB_PASSWORD=your_mysql_password
DB_NAME=your_database

🧮 MySQL Schema
racedates (holds dates to scrape)

CREATE TABLE racedates (
  id INT AUTO_INCREMENT PRIMARY KEY,
  RaceDate DATE NOT NULL
);

race_results (scraped data)

CREATE TABLE race_results (
  id INT AUTO_INCREMENT PRIMARY KEY,
  raceDateId INT,
  race_date DATE,
  course VARCHAR(10),
  race_no INT,
  race_info VARCHAR(255),
  pla VARCHAR(10),
  horse_no VARCHAR(10),
  horse VARCHAR(100),
  jockey VARCHAR(100),
  trainer VARCHAR(100),
  act_wt VARCHAR(10),
  declared_wt VARCHAR(10),
  draw_no VARCHAR(10),
  lbw VARCHAR(10),
  running_position VARCHAR(50),
  finish_time VARCHAR(20),
  win_odds VARCHAR(10),
  url TEXT,
  FOREIGN KEY (raceDateId) REFERENCES race_days(id)
);

prediction

CREATE TABLE future_races (
    id INT PRIMARY KEY AUTO_INCREMENT,
    race_info VARCHAR(100),
    horse_no INT,
    horse VARCHAR(100),
    draw_no INT,
    act_wt INT,
    jockey VARCHAR(100),
    trainer VARCHAR(100),
    win_odds FLOAT,
    place_odds FLOAT,
    race_date DATE,
    course VARCHAR(50),
    race_no INT
);

CREATE TABLE race_predictions (
    race_date DATE,
    race_no INT,
    horse_no INT,
    horse VARCHAR(100),
    win_probability FLOAT,
    place_probability FLOAT
);


🚀 How to Run

insert dates to racedates
e.g
INSERT INTO racedates (racedate) VALUES
('2023-07-23'),
('2023-07-29'),
('2023-07-30'),
('2023-08-01'),
('2023-08-02');

Run the scraper:

python scraper.py
It will:

Load race dates from race_days WHERE id BETWEEN START_ID AND END_ID

Skip if START_ID <= MAX(raceDateId) in race_results

Insert new race result data into race_results

🛡 Safety Features
✅ Prevents duplicate inserts via raceDateId

✅ Only scrapes valid races:

ST → Races 1–11

HV → Races 1–9

✅ Skips entire course if Race 1 is missing

✅ Loads DB credentials securely via .env

After the dataset was completed 

Then we can do the Horse Racing Prediction

Overview

This project contains a Python script (horse_racing_prediction.py) that predicts horse racing outcomes (win and place probabilities) using historical race data and machine learning. The script uses an XGBoost classifier to model the probability of a horse winning or placing (top 3) in a race, based on features like jockey and trainer performance, horse history, and race characteristics. It connects to a MySQL database to load historical and future race data, performs feature engineering (e.g., jockey_trainer_win_rate), and saves predictions to a CSV file and a MySQL table. A histogram of the jockey_trainer_win_rate distribution is also generated.

Features

Data Loading: Retrieves historical (race_results) and future (future_races) race data from a MySQL database using SQLAlchemy.

Feature Engineering:

Parses race_info to extract race class, distance, and rating range.

Computes win and place rates for horses, jockeys, and trainers.

Calculates jockey_trainer_win_rate with improved logic (uses jockey_win_rate as a fallback for pairs with race_count <= 1).

Includes rolling averages (e.g., last 5 races) and course-specific metrics.

Modeling: Trains two XGBoost models (is_winner, is_placed) with ROC AUC evaluation (~0.868 for win, ~0.873 for place).

Outputs:

Saves predictions (win_probability, place_probability) to predictions.csv.

Saves predictions to the MySQL race_predictions table.

Generates a 30-bin histogram of jockey_trainer_win_rate as JSON.

Environment: Compatible with Python 3.9+, pandas==1.5.3, xgboost==1.7.6, sqlalchemy==1.4.52.

Requirements

Python: 3.9 or higher

Dependencies:

pip install numpy==1.23.5 pandas==1.5.3 xgboost==1.7.6 scikit-learn==1.2.2 scipy==1.10.1 python-dotenv pymysql sqlalchemy==1.4.52

1. 
python scrape_race_card.py

2.
python horse_racing_prediction.py

📌 Notes
Ensure ChromeDriver version matches your local Chrome browser

You can schedule runs with cron, Windows Task Scheduler, or Python schedule

📄 License
MIT — Free to use and modify.
