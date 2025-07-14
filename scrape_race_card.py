import os
import time
import pymysql
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Load credentials from .env
load_dotenv()
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

# Setup Chrome (headless optional)
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(), options=chrome_options)
wait = WebDriverWait(driver, 10)

# Input Parameters
race_date = "2025-07-16"
venue = "HV"
race_range = range(1, 12) if venue == "ST" else range(1, 10)



for race_no in race_range:
    url = f"https://bet.hkjc.com/en/racing/wp/{race_date}/{venue}/{race_no}"
    try:
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "race-card-tab-container")))

        # Extract race_info
        race_info = "N/A"
        try:
            caption = driver.find_element(By.CLASS_NAME, "caption")
            race_info = caption.text.strip()
        except NoSuchElementException:
            print(f"⚠️ No race_info for {race_date} {venue} R{race_no}")

        # Extract each horse row
        rows = driver.find_elements(By.CSS_SELECTOR, ".racingTable tbody tr")
        for row in rows:
            try:
                cols = row.find_elements(By.TAG_NAME, "td")
                if not cols or len(cols) < 10:
                    continue

                horse_no = int(cols[0].text.strip())
                horse = cols[1].text.strip()
                draw_no = int(cols[2].text.strip())
                act_wt = int(cols[3].text.strip())
                jockey = cols[4].text.strip()
                trainer = cols[5].text.strip()

                # Extract Win/Place odds
                try:
                    win_odds = float(cols[6].text.strip())
                except:
                    win_odds = None
                try:
                    place_odds = float(cols[7].text.strip())
                except:
                    place_odds = None

                # Insert into DB
                cursor.execute("""
                    INSERT INTO future_races 
                    (race_info, horse_no, horse, draw_no, act_wt, jockey, trainer, win_odds, place_odds, race_date, course, race_no)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (race_info, horse_no, horse, draw_no, act_wt, jockey, trainer, win_odds, place_odds, race_date, venue, race_no))

            except Exception as e:
                print(f"⚠️ Failed to process a row: {e}")
        print(f"✅ Finished {race_date} {venue} R{race_no}")
        
    except TimeoutException:
        print(driver.page_source[:1000])
        print(f"❌ Skipped {race_date} {venue} Race {race_no}: Page not available")

# Cleanup
conn.commit()
cursor.close()
conn.close()
driver.quit()
