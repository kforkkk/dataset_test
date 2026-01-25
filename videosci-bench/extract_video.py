from pytubefix import YouTube
from pytubefix.cli import on_progress
import pandas as pd
import json
import os
import time
import csv
from tqdm import tqdm
import re

data = pd.read_csv("data.csv")
output_csv = "prompt.csv"
pattern = r"(\d+)"
with open("/hsk/dataset_test/data_filtered.jsonl", "r") as f, \
    open(output_csv, "a+") as f_out:
    csv_writer = csv.writer(f_out)
    filterd_data = f.readlines()
    for items in tqdm(filterd_data, desc="Processing items"):
        time.sleep(5)
        items = json.loads(items)
        #print(items)
        vid = items["vid"]
        editing_prompt = items["prompt"] 
        expected_phenomenon = items["expected phenomenon"] 
        csv_writer.writerow([editing_prompt, expected_phenomenon,vid])
        
        for i,id in enumerate(data["Unique ID"]):   
            match = re.search(pattern, str(id))
            if match and int(id) == int(vid):
                print("Downloading: ", vid)
                print("url:",data["Source"][i])
                
                try:
                    os.makedirs(f"/hsk/dataset_test/video/data_{vid}",exist_ok=True)
                    yt = YouTube(
                        str(data["Source"][i]),
                        on_progress_callback=on_progress,
                    )
                    
                    ys = yt.streams.get_highest_resolution()
                    ys.download(output_path=f"/hsk/dataset_test/video/data_{vid}",filename=f"video_{vid}")
                    print("Downloaded: ", vid)
                except:
                    print("Error: ", vid)

print("Done")