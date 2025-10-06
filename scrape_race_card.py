
import os
import sys
import time
import re
import argparse
import logging
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict

import pymysql
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# -----------------------------
# Setup logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------
# Env & DB config
# -----------------------------
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "charset": "utf8mb4",
    "autocommit": False,
    "cursorclass": pymysql.cursors.DictCursor,
}

REQUIRED_ENV = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]

# -----------------------------
# Race setup
# -----------------------------
race_date = "2025-10-04"   
course = "ST"              # "ST" , "HV"
race_range = range(1, 12) if course == "ST" else range(1, 10)

def validate_env() -> None:
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# -----------------------------
# Parsing helpers
# -----------------------------
def normalize_race_info(info: str) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str], Optional[str]]:
    """
    Parse Class, Distance, Surface, Track, Going from race_info string.
    Returns (race_class, distance, surface, track, going)
    """
    race_class = distance = surface = track = going = None

    # Class or Group
    m = re.search(r'(Class|Group)\s*(\d+)', info, re.IGNORECASE)
    if m:
        race_class = f"{m.group(1).title()} {m.group(2)}"

    # Distance (e.g. 1200m)
    m = re.search(r'(\d{3,4})\s*m\b', info, re.IGNORECASE)
    if m:
        try:
            distance = int(m.group(1))
        except ValueError:
            distance = None

    # Surface
    m = re.search(r'\b(TURF|ALL\s*WEATHER|DIRT|AWT)\b', info, re.IGNORECASE)
    if m:
        surface = m.group(1).upper().replace(" ", "") if m.group(1).upper() != "ALL WEATHER" else "ALL WEATHER"

    # Track / Course layout (e.g. "B Course", "C+3 Course")
    m = re.search(r'((?:[“"])?(?:A|B|C)(?:\+?[\d])?\s*Course)', info, re.IGNORECASE)
    if m:
        track = m.group(1).replace('"', '').replace('“', '').strip().upper()

    # Going
    m = re.search(r'\b(GOOD TO FIRM|GOOD|YIELDING|FIRM|SOFT|WET FAST|SLOW|HEAVY)\b', info, re.IGNORECASE)
    if m:
        going = m.group(1).upper()

    return race_class, distance, surface, track, going

# -----------------------------
# Selenium helpers
# -----------------------------
def build_driver(headless: bool = True) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1366,768")
    options.add_argument("--lang=en-US,en;q=0.9,zh-HK;q=0.8")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(45)
    return driver

def polite_sleep(sec: float) -> None:
    time.sleep(sec)

@contextmanager
def managed_driver(headless: bool = True):
    driver = build_driver(headless=headless)
    try:
        yield driver
    finally:
        try:
            driver.quit()
        except Exception:
            pass

# -----------------------------
# DB helpers
# -----------------------------
@contextmanager
def mysql_conn():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass

def upsert_future_races(cursor, rows: List[Dict]):
    """
    Insert or update rows into future_races.
    Requires a UNIQUE KEY on (race_date, course, race_no, horse_no).
    """
    if not rows:
        return

    sql = """
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
    ON DUPLICATE KEY UPDATE
        race_info=VALUES(race_info),
        horse=VALUES(horse),
        draw_no=VALUES(draw_no),
        act_wt=VALUES(act_wt),
        jockey=VALUES(jockey),
        trainer=VALUES(trainer),
        win_odds=VALUES(win_odds),
        place_odds=VALUES(place_odds),
        race_class=VALUES(race_class),
        distance=VALUES(distance),
        surface=VALUES(surface),
        track=VALUES(track),
        going=VALUES(going)
    """
    cursor.executemany(sql, rows)

# -----------------------------
# Scrape one race
# -----------------------------
def scrape_race(driver, race_date: str, course: str, race_no: int) -> List[Dict]:
    url = f"https://bet.hkjc.com/en/racing/wp/{race_date}/{course}/{race_no}"
    logging.info(f"Fetching Race {race_no} {course} {race_date}: {url}")
    last_exception = None
    for attempt in range(1, 4):
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            polite_sleep(1.5)
            html = driver.page_source
            break
        except Exception as e:
            last_exception = e
            logging.warning(f"Attempt {attempt}/3 failed for Race {race_no}: {e}")
            polite_sleep(2 * attempt)
    else:
        raise RuntimeError(f"Failed to load race page after retries: {last_exception}")

    os.makedirs("pages", exist_ok=True)
    html_file = os.path.join("pages", f"{race_date}_{course}_R{race_no}.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

    soup = BeautifulSoup(html, "html.parser")

    race_info = "N/A"
    try:
        text = soup.get_text(separator="\n", strip=True)
        match = re.search(r"(Class|Group|Restricted|Griffin)[^\n]*?(?:\d{3,4}\s*m|[Mm])?[^\n]*", text, re.IGNORECASE)
        if match:
            race_info = match.group(0).strip()
    except Exception as e:
        logging.warning(f"race_info extraction failed: {e}")
    logging.info(f"race_info: {race_info}")

    race_class, distance, surface, track, going = normalize_race_info(race_info)
    logging.info(f"Normalized: class={race_class}, dist={distance}, surface={surface}, track={track}, going={going}")

    tables = soup.find_all("table")
    target_table = None
    for table in tables:
        headers = [th.get_text(strip=True).upper() for th in table.find_all("th")]
        if "WIN" in headers and "PLACE" in headers:
            target_table = table
            break
    if not target_table and tables:
        logging.warning("No Win/Place table found. Using first table as fallback.")
        target_table = tables[0]
    if not target_table:
        logging.error("No table found on page.")
        return []

    rows = target_table.find_all("tr")[1:]
    out: List[Dict] = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) < 6:
            continue

        horse_no_txt = cells[0].get_text(strip=True)
        if not horse_no_txt.isdigit():
            continue
        horse_no = int(horse_no_txt)

        horse_name = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        if horse_name.lower() == "field" or not horse_name:
            continue

        def to_int(s: str) -> Optional[int]:
            s = s.strip()
            return int(s) if s.isdigit() else None

        def to_float(s: str) -> Optional[float]:
            s = s.strip()
            try:
                if not s or s in {"-", "--"}:
                    return None
                return float(s)
            except ValueError:
                return None

        draw_no = to_int(cells[3].get_text()) if len(cells) > 3 else None
        act_wt = to_int(cells[4].get_text()) if len(cells) > 4 else None

        # --- NEW: handle claim in jockey name, e.g. "P N Wong (-10)"
        jockey_raw = cells[5].get_text(strip=True) if len(cells) > 5 else ""
        claim_val = None
        m = re.search(r"\(-(?P<claim>\d+)\)", jockey_raw)
        if m:
            try:
                claim_val = int(m.group("claim"))
            except ValueError:
                claim_val = None
            jockey = jockey_raw[:m.start()].strip()
            if act_wt is not None and claim_val is not None:
                # Deduct claim from carried weight
                act_wt = max(act_wt - claim_val, 0)
        else:
            jockey = jockey_raw
        # --- end NEW

        trainer = cells[6].get_text(strip=True) if len(cells) > 6 else ""
        win_odds = to_float(cells[7].get_text()) if len(cells) > 7 else None
        place_odds = to_float(cells[8].get_text()) if len(cells) > 8 else None

        out.append({
            "race_info": race_info,
            "horse_no": horse_no,
            "horse": horse_name,
            "draw_no": draw_no,
            "act_wt": act_wt,
            "jockey": jockey,
            "trainer": trainer,
            "win_odds": win_odds,
            "place_odds": place_odds,
            "race_date": race_date,
            "course": course,
            "race_no": race_no,
            "race_class": race_class,
            "distance": distance,
            "surface": surface,
            "track": track,
            "going": going,
        })

    logging.info(f"Extracted {len(out)} rows")
    return out

# -----------------------------
# CLI & main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Scrape HKJC race card and upsert to MySQL")
    p.add_argument("--date", default=os.getenv("RACE_DATE", ""), help="Race date YYYY-MM-DD (default: env RACE_DATE)")
    p.add_argument("--course", default=os.getenv("COURSE", "ST"), choices=["ST", "HV"], help="Course code ST/HV")
    p.add_argument("--races", default="", help="Race numbers, e.g. 1-11 or 1,2,3 (default: autodetect by course)")
    p.add_argument("--headless", action="store_true", help="Run Chrome headless")
    return p.parse_args()

def expand_races(course: str, races_arg: str) -> range:
    if races_arg:
        if "-" in races_arg:
            a, b = races_arg.split("-", 1)
            return range(int(a), int(b) + 1)
        else:
            nums = [int(x.strip()) for x in races_arg.split(",") if x.strip().isdigit()]
            return range(min(nums), max(nums) + 1) if nums else range(1, 1)
    return range(1, 12) if course == "ST" else range(1, 10)

def main():
    validate_env()
    logging.info(f"Scraping {course} on {race_date}, races {race_range.start}-{race_range.stop - 1}")

    with managed_driver(headless=True) as driver, mysql_conn() as conn:
        cursor = conn.cursor()
        total_rows = 0
        for race_no in race_range:
            try:
                rows = scrape_race(driver, race_date, course, race_no)
                if rows:
                    upsert_future_races(cursor, rows)
                    conn.commit()
                    total_rows += len(rows)
                    logging.info(f"Upserted {len(rows)} rows for Race {race_no}")
                else:
                    logging.warning(f"No rows extracted for Race {race_no}")
            except Exception as e:
                conn.rollback()
                logging.exception(f"Error processing Race {race_no}: {e}")
                continue

        logging.info(f"Done. Total rows upserted: {total_rows}")

if __name__ == "__main__":
    main()
