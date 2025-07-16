Horse Racing Prediction Pipeline (HKJC)
Overview
This project provides an end-to-end pipeline for scraping Hong Kong Jockey Club (HKJC) horse racing data, storing it in MySQL, engineering features, training prediction models (XGBoost), and outputting probability-based forecasts (including Monte Carlo simulations) for upcoming races.

Folder Structure
bash
Copy
Edit
.
├── scrape_race_results.py      # Scrape past race results into MySQL
├── scrape_race_card.py         # Scrape future race cards (race entries & odds)
├── horse_racing_prediction.py  # Feature engineering, model training, prediction & simulation
├── tables.sql                  # SQL file to create all necessary tables
├── .env                        # Environment variables (DB credentials etc.)
└── README.md                   # This file
Prerequisites
Python 3.8+

MySQL server

Chrome browser and ChromeDriver (for Selenium scraping)

Python libraries:

selenium

pymysql

sqlalchemy

python-dotenv

pandas

numpy

xgboost

tqdm

beautifulsoup4

Install required libraries:

bash
Copy
Edit
pip install selenium pymysql sqlalchemy python-dotenv pandas numpy xgboost tqdm beautifulsoup4
Setup
Database

Run tables.sql in your MySQL server to create all necessary tables:

bash
Copy
Edit
mysql -u youruser -p yourdb < tables.sql
Environment Variables

Create a .env file in your project folder with:

ini
Copy
Edit
DB_HOST=localhost
DB_USER=your_mysql_user
DB_PASSWORD=your_password
DB_NAME=your_database_name
ChromeDriver

Download the version matching your Chrome from here

Ensure it’s on your system PATH or specify its path in your scripts if needed.

Usage
1. Scrape Historical Race Results
bash
Copy
Edit
python scrape_race_results.py
Scrapes past results from HKJC and inserts them into the race_results table.

Uses Selenium to automate data extraction for each race date.

2. Scrape Future Race Cards
bash
Copy
Edit
python scrape_race_card.py
Scrapes entries and odds for upcoming races into the future_races table.

Designed to be run before model prediction (preferably race day).

3. Run Predictions & Monte Carlo Simulations
bash
Copy
Edit
python horse_racing_prediction.py
Does everything: Feature engineering, model training (win/place), prediction, and Monte Carlo simulation for win/place.

Outputs predictions to race_predictions table in MySQL and optionally to a CSV.

Model uses the latest 400 racedays (or all if fewer).

Table Overview
racedates: List of historical race days.

race_results: Historical race results and metadata.

future_races: Runners and odds for upcoming races.

race_predictions: Final output with model and simulated probabilities for win/place.

race_predictions columns:
Column	Description
race_date	Race date (datetime)
race_no	Race number
horse_no	Horse number
horse	Horse name
win_probability	Model-predicted win probability
place_probability	Model-predicted place probability
sim_win_pct	Simulated win % (Monte Carlo)
sim_place_pct	Simulated place % (Monte Carlo)

Notes
All code is modular: You can update, rerun, or adapt any stage independently.

Extensible: Add more features, models, or export destinations as needed.

Safe for repeated runs: Insertions are handled robustly; model predictions always reflect the most current data.

Credits
Developed by Daniel Chan and ChatGPT (OpenAI) for Hong Kong racing analytics.

Troubleshooting
Make sure .env is filled correctly and MySQL is running.

If Selenium or ChromeDriver throws an error, check versions or PATH.

For very large databases, adjust the script queries or table sizes as needed.