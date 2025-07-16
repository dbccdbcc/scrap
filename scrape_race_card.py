import os
import sys
import time
import re
import pymysql
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DB config
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": "utf8mb4"
}

# Normalization function
def normalize_race_info(info):
    """
    Parse Class, Distance, Surface, Track, Going from race_info string.
    Returns (race_class, distance, surface, track, going)
    """
    race_class = distance = surface = track = going = None

    # Parse for "Class 5, 1200m, TURF, ... , 'B' Course, GOOD TO FIRM"
    m = re.search(r'Class\s*(\d+)', info, re.IGNORECASE)
    if m:
        race_class = f"Class {m.group(1)}"
    m = re.search(r'(\d{3,4})\s*m', info, re.IGNORECASE)
    if m:
        distance = int(m.group(1))
    m = re.search(r'(TURF|ALL WEATHER|DIRT)', info, re.IGNORECASE)
    if m:
        surface = m.group(1).upper()
    m = re.search(r'(["“”]?[ABCD][\s"]*Course)', info, re.IGNORECASE)
    if m:
        track = m.group(1).replace('"', '').strip()
    m = re.search(r'(GOOD TO FIRM|GOOD|YIELDING|FIRM|SOFT|WET FAST|SLOW|HEAVY)', info, re.IGNORECASE)
    if m:
        going = m.group(1).upper()
    return race_class, distance, surface, track, going

# Race setup
race_date = "2025-07-16"
course = "HV"
race_range = range(1, 12) if course == "ST" else range(1, 10)

# Set up Selenium
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

try:
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    for race_no in race_range:
        url = f"https://bet.hkjc.com/en/racing/wp/{race_date}/{course}/{race_no}"
        print(f"\n🔍 Processing Race {race_no} at {course}...")

        try:
            driver.get(url)
            time.sleep(3)
            html = driver.page_source

            html_file = f"temp_race_R{race_no}.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html)
            print("✅ Webpage saved.")
        except Exception as e:
            print(f"❌ Error loading Race {race_no}: {e}")
            continue

        # Parse with BeautifulSoup
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Get race_info
        race_info = "N/A"
        try:
            text = soup.get_text(separator="\n", strip=True)
            match = re.search(r"(Class|Group|Restricted|Griffin)[^\n]*?(?:\d{3,4}|M)[^\n]*", text, re.IGNORECASE)
            if match:
                race_info = match.group(0).strip()
        except Exception as e:
            print(f"⚠️ race_info extraction failed: {e}")
        print(f"✅ race_info: {race_info}")

        # Parse out normalized info
        race_class, distance, surface, track, going = normalize_race_info(race_info)
        print(f"🧩 Normalized: class={race_class}, dist={distance}, surface={surface}, track={track}, going={going}")

        # Find race table
        tables = soup.find_all("table")
        print(f"✅ Found {len(tables)} table(s)")

        target_table = None
        for table in tables:
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            if "Win" in headers and "Place" in headers:
                target_table = table
                break

        if not target_table:
            if tables:
                target_table = tables[0]
                print("⚠️ No Win/Place table found. Using fallback first table.")
            else:
                print("❌ No table found.")
                continue

        # Extract data
        rows = target_table.find_all("tr")[1:]
        extracted = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 7:
                continue

            try:
                horse_no = int(cells[0].text.strip())
            except ValueError:
                continue  # Skip Field/invalid

            horse_name = cells[2].text.strip()
            if horse_name.lower() == "field":
                continue

            extracted.append({
                "race_info": race_info,
                "horse_no": horse_no,
                "horse": horse_name,
                "draw_no": int(cells[3].text.strip()) if cells[3].text.strip().isdigit() else None,
                "act_wt": int(cells[4].text.strip()) if cells[4].text.strip().isdigit() else None,
                "jockey": cells[5].text.strip(),
                "trainer": cells[6].text.strip(),
                "win_odds": float(cells[7].text.strip()) if len(cells) > 7 and cells[7].text.strip().replace('.', '', 1).isdigit() else None,
                "place_odds": float(cells[8].text.strip()) if len(cells) > 8 and cells[8].text.strip().replace('.', '', 1).isdigit() else None,
                "race_date": race_date,
                "course": course,
                "race_no": race_no,
                "race_class": race_class,
                "distance": distance,
                "surface": surface,
                "track": track,
                "going": going
            })

        print(f"✅ Extracted {len(extracted)} rows")

        if extracted:
            insert_sql = """
                INSERT INTO future_races (
                    race_info, horse_no, horse, draw_no, act_wt,
                    jockey, trainer, win_odds, place_odds,
                    race_date, course, race_no,
                    race_class, distance, surface, track, going
                ) VALUES (
                    %(race_info)s, %(horse_no)s, %(horse)s, %(draw_no)s, %(act_wt)s,
                    %(jockey)s, %(trainer)s, %(win_odds)s, %(place_odds)s,
                    %(race_date)s, %(course)s, %(race_no)s,
                    %(race_class)s, %(distance)s, %(surface)s, %(track)s, %(going)s
                )
            """
            try:
                cursor.executemany(insert_sql, extracted)
                conn.commit()
                print(f"✅ Inserted {len(extracted)} rows into future_races")
            except Exception as db_error:
                print(f"❌ DB error: {db_error}")
        else:
            print("⚠️ No data to insert.")

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
    driver.quit()
