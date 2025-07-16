# Horse Racing Prediction Pipeline (HKJC)

## Overview

This project provides a full pipeline for:
- **Scraping Hong Kong Jockey Club (HKJC) horse racing data**
- **Storing structured data in MySQL**
- **Feature engineering & model training (XGBoost)**
- **Predicting win/place probabilities and running Monte Carlo simulations for future races**

---

## Folder Structure

```
.
├── scrape_race_results.py      # Scrapes past race results to MySQL
├── scrape_race_card.py         # Scrapes future race cards (entries & odds)
├── horse_racing_prediction.py  # Feature engineering, model training, prediction & simulation
├── tables.sql                  # SQL for all necessary tables
├── .env                        # Environment variables (DB credentials)
└── README.md                   # This file
```

---

## Prerequisites

- **Python 3.8+**
- **MySQL Server**
- [ChromeDriver](https://chromedriver.chromium.org/downloads) & Chrome browser (for Selenium scraping)
- Python packages:
  - `selenium`
  - `pymysql`
  - `sqlalchemy`
  - `python-dotenv`
  - `pandas`
  - `numpy`
  - `xgboost`
  - `tqdm`
  - `beautifulsoup4`

Install requirements:
```bash
pip install selenium pymysql sqlalchemy python-dotenv pandas numpy xgboost tqdm beautifulsoup4
```

---

## Setup

1. **Create Database Tables**

   - Run `tables.sql` in MySQL:
     ```bash
     mysql -u youruser -p yourdb < tables.sql
     ```

2. **.env file**

   - Place your DB credentials in `.env`:
     ```
     DB_HOST=localhost
     DB_USER=your_mysql_user
     DB_PASSWORD=your_password
     DB_NAME=your_database_name
     ```

3. **ChromeDriver**

   - Download ChromeDriver matching your Chrome browser version [here](https://chromedriver.chromium.org/downloads).
   - Add ChromeDriver to your PATH or set its path in your scripts.

---

## Usage

### 1. **Scrape Historical Race Results**

```bash
python scrape_race_results.py
```
- Scrapes and inserts past race results into the `race_results` table.

### 2. **Scrape Future Race Cards**

```bash
python scrape_race_card.py
```
- Scrapes entries and odds for upcoming races into the `future_races` table.

### 3. **Run Model Prediction & Monte Carlo Simulation**

```bash
python horse_racing_prediction.py
```
- Does feature engineering, trains XGBoost models (win/place), predicts for future races, and runs Monte Carlo simulations.
- Results are written to `race_predictions` (MySQL) and `race_predictions_output.csv`.

---

## Table Overview

- **`racedates`**: List of all race days.
- **`race_results`**: Historical results and runner details.
- **`future_races`**: Runners and odds for upcoming races.
- **`race_predictions`**: Model & simulated win/place probabilities for future races.

### `race_predictions` columns

| Column             | Description                                   |
|--------------------|-----------------------------------------------|
| race_date          | Race date (datetime)                          |
| race_no            | Race number                                   |
| horse_no           | Horse number                                  |
| horse              | Horse name                                    |
| win_probability    | Model-predicted win probability               |
| place_probability  | Model-predicted place probability             |
| sim_win_pct        | Simulated win % (Monte Carlo)                 |
| sim_place_pct      | Simulated place % (Monte Carlo)               |

---

## Notes

- **Modular:** Each stage can be updated or rerun independently.
- **Extensible:** Add more features, models, or outputs as desired.
- **Safe for repeated runs:** Database updates are robust; predictions always reflect current data.

---

## Troubleshooting

- Make sure `.env` is filled and MySQL is running.
- Ensure ChromeDriver matches your Chrome version.
- For large databases, you may need to adjust script queries or memory.

---

**Developed by Daniel Chan**

Happy racing! 🏇
