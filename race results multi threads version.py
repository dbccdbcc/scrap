"""
@author: Daniel
"""
import os
import math
import time
import random
import gc
from multiprocessing import Process
from dotenv import load_dotenv
import pymysql
import re
from bs4 import BeautifulSoup

# BONUS: Only if you have psutil installed; otherwise, comment out.
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def parse_race_conditions(soup):
    # ... (same as before, omitted for brevity)
    text = soup.get_text(separator="\n", strip=True)
    going = None
    surface = None
    track = None
    going_match = re.search(r"Going\s*:\s*([^\n]+)", text, re.IGNORECASE)
    if going_match:
        going = going_match.group(1).strip().upper()
    course_match = re.search(r"Course\s*:\s*([^\n]+)", text, re.IGNORECASE)
    if course_match:
        course_field = course_match.group(1).strip().upper()
        if "ALL WEATHER" in course_field:
            surface = "ALL WEATHER TRACK"
            track = "ALL WEATHER TRACK"
        elif "TURF" in course_field:
            surface = "TURF"
            m = re.search(r'"([A-E](?:\+\d)?)"\s*COURSE', course_field)
            if m:
                track = m.group(1)
        else:
            surface = course_field
    if surface == "TURF" and track is None:
        m2 = re.search(r'"([A-E](?:\+\d)?)"\s*COURSE', text)
        if m2:
            track = m2.group(1)
    return surface, track, going

def parse_race_info(race_info):
    # ... (same as before)
    race_class = None
    distance = None
    if '-' in race_info:
        race_class = race_info.split('-')[0].strip()
    else:
        race_class = race_info.strip()
    dist_match = re.search(r'(\d{3,4})\s*M', race_info)
    if dist_match:
        distance = int(dist_match.group(1))
    return race_class, distance

def cleanup_chrome_driver(driver):
    """Ensure Chrome and chromedriver are fully closed, with bonus zombie cleanup."""
    try:
        driver.quit()
    except Exception:
        pass
    # Bonus: Use psutil to kill stray chrome/chromedriver if any.
    if PSUTIL_AVAILABLE:
        proc_names = ['chromedriver.exe', 'chrome.exe']
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] in proc_names:
                try:
                    proc.terminate()
                except Exception:
                    pass

def print_mem(prefix=""):
    # Print memory usage of current process for debug.
    import os, psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"{prefix} Memory: {mem:.1f} MB")

def scrape_batch(race_dates_batch, batch_no):
    import pymysql
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
    from bs4 import BeautifulSoup
    import gc
    import time
    import traceback

    print(f"Batch {batch_no} started. Total race dates: {len(race_dates_batch)}")
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

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument('--log-level=3')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    # Small random sleep to stagger process start and reduce initial Chrome lag
    time.sleep(random.uniform(0.5, 2.0))

    try:
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 30)
        date_remain = len(race_dates_batch)
        for entry in race_dates_batch:
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
                except TimeoutException:
                    print(f"Batch {batch_no}: ❌ {date_str} {course} R1 not found (Timeout), skip course")
                    continue
                except WebDriverException as e:
                    print(f"Batch {batch_no}: ❌ {date_str} {course} R1 webdriver error, skip course: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Batch {batch_no}: ❌ {date_str} {course} R1 unknown error, skip course: {str(e)}")
                    continue

                if race_exists:
                    print(f"{date_remain} remaining")
                    date_remain = date_remain - 1
                    print(f"Batch {batch_no}: 🔁 {date_str} {course} R1 exists, checking races 1–11")
                    race_range = range(1, 12) if course == "ST" else range(1, 10)
                    for idx, race_no in enumerate(race_range, 1):
                        url = f"https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={date_str}&Racecourse={course}&RaceNo={race_no}"
                        try:
                            t0 = time.time()
                            driver.get(url)
                            wait.until(EC.presence_of_element_located((By.XPATH, "//table[.//td[contains(text(),'Pla.')]]")))
                            html = driver.page_source
                            soup = BeautifulSoup(html, "html.parser")
                            surface, track, going = parse_race_conditions(soup)

                            race_info = "N/A"
                            try:
                                td_elements = driver.find_elements(By.XPATH, "//td")
                                for td in td_elements:
                                    lines = td.text.strip().splitlines()
                                    for line in lines:
                                        if ("Class" in line or "Group" in line or "Restricted" in line or "Griffin" in line or "Hong" in line) and "M" in line:
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
                                min_expected_cols = 12
                                cell_texts = [cell.text.strip() for cell in cells]
                                while len(cell_texts) < min_expected_cols:
                                    cell_texts.append("N/A")

                                data = {
                                    "Date": date_str,
                                    "Course": course,
                                    "RaceNo": race_no,
                                    "RaceInfo": race_info,
                                    "Pla": cell_texts[0],
                                    "HorseNo": cell_texts[1],
                                    "Horse": cell_texts[2],
                                    "Jockey": cell_texts[3],
                                    "Trainer": cell_texts[4],
                                    "ActWt": cell_texts[5],
                                    "DeclaredWt": cell_texts[6],
                                    "Draw": cell_texts[7],
                                    "LBW": cell_texts[8],
                                    "RunningPosition": cell_texts[9],
                                    "FinishTime": cell_texts[10],
                                    "WinOdds": cell_texts[11],
                                    "URL": url,
                                    "RaceDateId": date_id,
                                    "Race_class": race_class,
                                    "Distance": distance,
                                    "Surface": surface,
                                    "Track": track,
                                    "Going": going
                                }

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
                                    data["Race_class"], data["Distance"], data["Surface"], data["Track"], data["Going"]
                                )
                                try:
                                    cursor.execute(sql, values)
                                except Exception as ex:
                                    print(f"Batch {batch_no}: ❌ DB Insert error on {date_str} {course} R{race_no} row: {cell_texts}")
                                    print(f"    SQL values: {values}")
                                    print(f"    Exception: {ex}")

                            # Progress print for long runs
                            if idx % 3 == 0:
                                print(f"Batch {batch_no}: Progress {date_str} {course} R{race_no} ({idx}/{len(race_range)}) | Elapsed: {round(time.time()-t0,1)}s")

                            # Extra clean-up
                            del rows, table, td_elements
                            if idx % 3 == 0:
                                gc.collect()
                            #slow down a bit to avoid browser queue congestion
                            time.sleep(random.uniform(0.1, 0.3))
                            #print_mem(f"Batch {batch_no} {date_str} {course} R{race_no}:") #for memory monitoring
                            print(f"Batch {batch_no}: ✅ {date_str} {course} R{race_no} extracted and inserted")
                        except TimeoutException:
                            print(f"Batch {batch_no}: ⚠️ Timeout: {date_str} {course} R{race_no}, skipping.")
                        except Exception as ex:
                            #print(f"Batch {batch_no}: ⚠️ Skipped: {date_str} {course} R{race_no}, {str(ex)}")
                            print(f"Batch {batch_no}: ⚠️ Skipped: {date_str} {course} R{race_no}")
                            #print(traceback.format_exc())
                # Full GC at the end of each date
                gc.collect()
        conn.commit()
    except Exception as e:
        print(f"Batch {batch_no} error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        cleanup_chrome_driver(driver)
        print(f"Batch {batch_no}: ChromeDriver closed.")
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass
        print(f"Batch {batch_no} completed.")

if __name__ == "__main__":
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
    cursor.execute("SELECT MAX(raceDateId) FROM race_results")
    max_id_in_results = cursor.fetchone()[0] or 0
    START_ID = max_id_in_results + 1
    END_ID = START_ID + 19  # or as needed, not too much, will cause chrome drive crash
    cursor.execute(
        "SELECT id, RaceDate FROM racedates WHERE id BETWEEN %s AND %s ORDER BY id",
        (START_ID, END_ID)
    )
    race_dates = [
        {"id": row[0], "RaceDate": row[1].strftime("%Y/%m/%d") if hasattr(row[1], 'strftime') else row[1]}
        for row in cursor.fetchall()
    ]
    cursor.close()
    conn.close()
    num_batches = 2  # Adjust for your hardware. if 3 or more do not scrap more than 100 days
    batch_size = math.ceil(len(race_dates) / num_batches)
    processes = []
    for i in range(num_batches):
        batch = race_dates[i*batch_size : (i+1)*batch_size]
        print(f"Batch {i+1}: {batch}")
        p = Process(target=scrape_batch, args=(batch, i+1))
        processes.append(p)
        p.start()
        # Stagger process start to avoid Chrome congestion
        time.sleep(random.uniform(0.2, 0.6))
    for p in processes:
        p.join()
    print("✅ All batches completed.")
