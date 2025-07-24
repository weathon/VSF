from datasets import load_dataset
import random
import os
import shutil
import subprocess


subprocess.run(['python3', 'wan_vsf.py'])
subprocess.run(['python3', 'wan_nag.py'])
subprocess.run(['python3', 'wan_none.py'])