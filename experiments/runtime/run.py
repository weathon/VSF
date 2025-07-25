from datasets import load_dataset
import random
import os
import shutil
import subprocess

files = os.listdir(".")
files = [f for f in files if f.endswith(".py") and f != "run.py"]

import time
for f in files:
    subprocess.run(['python3', f])
    time.sleep(60 * 5)