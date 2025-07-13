# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 18:28:41 2025

@author: User
"""
import os
from dotenv import load_dotenv
import pymysql
import pandas as pd

load_dotenv()
print("User =", os.getenv("DB_USER"))
print("Password =", os.getenv("DB_PASSWORD"))

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