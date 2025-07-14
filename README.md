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

for future development (prediction)

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

📌 Notes
Ensure ChromeDriver version matches your local Chrome browser

You can schedule runs with cron, Windows Task Scheduler, or Python schedule

📄 License
MIT — Free to use and modify.
