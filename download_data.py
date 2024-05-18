import numpy as np
import pandas as pd 
import kaggle
import json
import os
import subprocess
ddir = kaggle.api.get_default_download_dir()
print(ddir)
file_lst = kaggle.api.dataset_list_files("viktoriiashkurenko/278k-spotify-songs/")
file_lst.files[0]
if not os.path.exists("data"):
    os.mkdir("data")
for file in file_lst.files:
    kaggle.api.dataset_download_file("viktoriiashkurenko/278k-spotify-songs", str(file), path=os.getcwd() + "/data", force=True)
for file in os.listdir("data"):
    if file.endswith(".zip"):
        subprocess.run(["unzip", os.path.join("data", file), "-d", "data"])
        #os.remove(os.path.join("data", file))