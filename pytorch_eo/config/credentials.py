from pathlib import Path
import requests
import json
from .config import SCAN_URL
import time
import os


def retrieve_credentials():
    creds_path = Path.home() / ".ep/creds.json"
    try:
        with open(creds_path) as json_file:
            creds = json.load(json_file)
    except:
        login_url = requests.get(f"{SCAN_URL}/auth/login").json()
        print(f"Login at: {login_url} and then enter your code.")
        time.sleep(0.5)
        id_token = input(f"Code:")
        creds = {"id_token": id_token}
        os.makedirs(Path.home() / ".ep", exist_ok=True)
        with open(creds_path, "w") as outfile:
            json.dump(creds, outfile)
    return creds
