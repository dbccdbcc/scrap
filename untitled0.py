# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 18:28:41 2025

@author: User
"""
from dotenv import load_dotenv
import os

load_dotenv()
print("User =", os.getenv("DB_USER"))
print("Password =", os.getenv("DB_PASSWORD"))