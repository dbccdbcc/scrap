# -*- coding: utf-8 -*-
"""
@author: Daniel
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
from dotenv import load_dotenv
import pymysql
import pandas as pd


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



# Get race dates from MySQL table
START_ID = 0
END_ID = 0


# Get the highest raceDateId already in race_results
cursor.execute("SELECT MAX(raceDateId) FROM race_results")
max_id_in_results = cursor.fetchone()[0] or 0  # fallback to 0 if None

START_ID = max_id_in_results + 1
END_ID = START_ID + 4  # or any batch size


# Safety check
if START_ID <= max_id_in_results:
    raise ValueError(
        f"❌ START_ID ({START_ID}) must be greater than the max raceDateId already in race_results ({max_id_in_results})"
    )
    
cursor.execute(
    "SELECT id, RaceDate FROM racedates WHERE id BETWEEN %s AND %s ORDER BY id",
    (START_ID, END_ID)
)
race_dates = [
    {"id": row[0], "RaceDate": row[1].strftime("%Y/%m/%d") if hasattr(row[1], 'strftime') else row[1]}
    for row in cursor.fetchall()
]


# Setup Selenium
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)

# Load dates
df_dates = pd.read_excel("RaceDateList.xlsx")

for entry in race_dates:
    date_id = entry["id"]
    date_str = entry["RaceDate"]
#for date_str in ["2010/01/01"]:
    for course in ["ST", "HV"]:
        race_exists = False

        # Check if Race 1 exists
        race_no = 1
        url = f"https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={date_str}&Racecourse={course}&RaceNo={race_no}"

        try:
            driver.get(url)
            wait.until(EC.presence_of_element_located((By.XPATH, "//table[.//td[contains(text(),'Pla.')]]")))
            race_exists = True
        except Exception:
            print(f"❌ {date_str} {course} R1 not found, skip course")
            continue

        if race_exists:
            print(f"🔁 {date_str} {course} R1 exists, checking races 1–11")
            race_range = range(1, 12) if course == "ST" else range(1, 10)
            
            for race_no in race_range:
                url = f"https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={date_str}&Racecourse={course}&RaceNo={race_no}"
                try:
                    driver.get(url)
                    wait.until(EC.presence_of_element_located((By.XPATH, "//table[.//td[contains(text(),'Pla.')]]")))

                    # Extract race info (Class or Group line)
                    race_info = "N/A"
                    try:
                        td_elements = driver.find_elements(By.XPATH, "//td")
                        for td in td_elements:
                            lines = td.text.strip().splitlines()
                            for line in lines:
                                if ("Class" in line or "Group" in line) and "M" in line:
                                    race_info = line.strip()
                                    raise StopIteration
                    except StopIteration:
                        pass
                    except NoSuchElementException:
                        pass

                    # Extract all horse rows
                    table = driver.find_element(By.XPATH, "//table[.//td[contains(text(),'Pla.')]]")
                    rows = table.find_elements(By.TAG_NAME, "tr")[1:]

                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) < 12:
                            continue

                        data = {
                            "Date": date_str,
                            "Course": course,
                            "RaceNo": race_no,
                            "RaceInfo": race_info,
                            "Pla": cells[0].text.strip(),
                            "HorseNo": cells[1].text.strip(),
                            "Horse": cells[2].text.strip(),
                            "Jockey": cells[3].text.strip(),
                            "Trainer": cells[4].text.strip(),
                            "ActWt": cells[5].text.strip(),
                            "DeclaredWt": cells[6].text.strip(),
                            "Draw": cells[7].text.strip(),
                            "LBW": cells[8].text.strip(),
                            "RunningPosition": cells[9].text.strip(),
                            "FinishTime": cells[10].text.strip(),
                            "WinOdds": cells[11].text.strip(),
                            "URL": url,
                            "RaceDateId": date_id
                        }

                        # Insert into MySQL
                        sql = """
                            INSERT INTO race_results (
                                race_date, course, race_no, race_info,
                                pla, horse_no, horse, jockey, trainer,
                                act_wt, declared_wt, draw_no, lbw,
                                running_position, finish_time, win_odds, url,raceDateId
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        values = (
                            data["Date"], data["Course"], data["RaceNo"], data["RaceInfo"],
                            data["Pla"], data["HorseNo"], data["Horse"], data["Jockey"], data["Trainer"],
                            data["ActWt"], data["DeclaredWt"], data["Draw"], data["LBW"],
                            data["RunningPosition"], data["FinishTime"], data["WinOdds"], data["URL"],data["RaceDateId"]
                        )
                        cursor.execute(sql, values)

                    print(f"✅ {date_str} {course} R{race_no} extracted and inserted")

                except Exception:
                    print(f"⚠️ Skipped: {date_str} {course} R{race_no}")

# Finalize
conn.commit()
cursor.close()
conn.close()
driver.quit()
print("✅ All data inserted into MySQL successfully.")