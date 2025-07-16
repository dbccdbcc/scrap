"""
@author: Daniel
"""
import gc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import os
from dotenv import load_dotenv
import pymysql
from bs4 import BeautifulSoup
import re

# -- Environment and DB Setup --
load_dotenv()
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": "utf8mb4"
}
conn = pymysql.connect(**db_config)
cursor = conn.cursor()

# Get race dates from MySQL table
cursor.execute("SELECT MAX(raceDateId) FROM race_results")
max_id_in_results = cursor.fetchone()[0] or 0  # fallback to 0 if None
START_ID = max_id_in_results + 1
END_ID = START_ID + 5  # or any batch size
dayRemaining = END_ID-START_ID+1

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

options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

# --- New function for parsing surface/track/going from HTML ---
def parse_race_conditions(soup):
    text = soup.get_text(separator="\n", strip=True)
    going = None
    surface = None
    track = None

    # Extract Going
    going_match = re.search(r"Going\s*:\s*([^\n]+)", text, re.IGNORECASE)
    if going_match:
        going = going_match.group(1).strip().upper()

    # Extract Surface and Track from 'Course : ...'
    course_match = re.search(r"Course\s*:\s*([^\n]+)", text, re.IGNORECASE)
    if course_match:
        course_field = course_match.group(1).strip().upper()
        # Surface
        if "ALL WEATHER" in course_field:
            surface = "ALL WEATHER TRACK"
            track = "ALL WEATHER TRACK"
        elif "TURF" in course_field:
            surface = "TURF"
            # Look for track pattern like "C+3", "A", "B", "A+2", etc.
            m = re.search(r'"([A-E](?:\+\d)?)"\s*COURSE', course_field)
            if m:
                track = m.group(1)
        else:
            surface = course_field

    # As a fallback, scan the whole text for '"C+3" COURSE' etc
    if surface == "TURF" and track is None:
        m2 = re.search(r'"([A-E](?:\+\d)?)"\s*COURSE', text)
        if m2:
            track = m2.group(1)

    return surface, track, going

def parse_race_info(race_info):
    race_class = None
    distance = None

    # Extract class
    class_match = re.match(r'(Class \d+|Group \d+|Restricted Race|Griffin Race)', race_info, re.IGNORECASE)
    if class_match:
        race_class = class_match.group(1).title()
    else:
        # If not matching the above, try capturing text before the first dash as fallback
        fallback = race_info.split('-')[0].strip()
        if fallback:
            race_class = fallback.title()

    # Extract distance (allow for no space before M)
    dist_match = re.search(r'(\d{3,4})\s*M', race_info)
    if dist_match:
        distance = int(dist_match.group(1))

    return race_class, distance


for entry in race_dates:
    date_id = entry["id"]
    date_str = entry["RaceDate"]
    for course in ["ST", "HV"]:
        race_exists = False
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
            print(f"{dayRemaining} day left")
            dayRemaining = dayRemaining - 1
            for race_no in race_range:
                url = f"https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={date_str}&Racecourse={course}&RaceNo={race_no}"
                try:
                    driver.get(url)
                    wait.until(EC.presence_of_element_located((By.XPATH, "//table[.//td[contains(text(),'Pla.')]]")))

                    # --- Parse race condition fields from HTML ---
                    html = driver.page_source
                    soup = BeautifulSoup(html, "html.parser")
                    surface, track, going = parse_race_conditions(soup)
                    
                    # Extract race info (Class or Group line)
                    race_info = "N/A"
                    try:
                        td_elements = driver.find_elements(By.XPATH, "//td")
                        for td in td_elements:
                            lines = td.text.strip().splitlines()
                            for line in lines:
                                if ("Class" in line or "Group" in line or "Restricted" in line or "Griffin" in line) and "M" in line:
                                    race_info = line.strip()
                                    raise StopIteration
                    except StopIteration:
                        pass
                    except NoSuchElementException:
                        pass
                    race_class, distance = parse_race_info(race_info)
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
                            "RaceDateId": date_id,
                            "Race_class": race_class,
                            "Distance": distance,
                            "Surface": surface,
                            "Track": track,
                            "Going": going
                        }

                        # Insert into MySQL
                        sql = """
                            INSERT INTO race_results (
                                race_date, course, race_no, race_info,
                                pla, horse_no, horse, jockey, trainer,
                                act_wt, declared_wt, draw_no, lbw,
                                running_position, finish_time, win_odds, url, raceDateId,
                                race_class, distance, surface, track, going
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        values = (
                            data["Date"], data["Course"], data["RaceNo"], data["RaceInfo"],
                            data["Pla"], data["HorseNo"], data["Horse"], data["Jockey"], data["Trainer"],
                            data["ActWt"], data["DeclaredWt"], data["Draw"], data["LBW"],
                            data["RunningPosition"], data["FinishTime"], data["WinOdds"], data["URL"], data["RaceDateId"],
                            data["Race_class"],data["Distance"],data["Surface"], data["Track"], data["Going"]
                        )
                        cursor.execute(sql, values)

                    print(f"✅ {date_str} {course} R{race_no} extracted and inserted")
                    try:
                        del rows
                    except: pass
                    try:
                        del table
                    except: pass
                    try:
                        del td_elements
                    except: pass
                    gc.collect()
                except Exception:
                    print(f"⚠️ Skipped: {date_str} {course} R{race_no}")

# Finalize
conn.commit()
cursor.close()
conn.close()
driver.quit()
print("✅ All data inserted into MySQL successfully.")
