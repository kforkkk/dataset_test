import pandas as pd
import os
import json

df = pd.read_parquet('/hsk/dataset_test/train-00000-of-00001.parquet')
#print(df.columns)
subcate_dict = {}
subcate_data = {}
for i,item in enumerate(df["category"]):
    if "Anti-Physics" in item:
        subcategory = df["subcategory"][i]
        os.makedirs(f"/hsk/dataset_test/anti-physic/{subcategory}",exist_ok=True) 
        prompt = df["Prompt"][i]
        if subcate_dict.get(subcategory) is None:
            subcate_dict[subcategory] = 1
            subcate_data[subcategory] = [{"id": 1, "prompt": prompt}]
        else:
            subcate_dict[subcategory] += 1
            subcate_data[subcategory].append({"id": subcate_dict[subcategory], "prompt": prompt})
            
for key in subcate_data.keys():
    with open(f"/hsk/dataset_test/anti-physic/{key}/{key}.jsonl","w") as f:
        for item in subcate_data[key]:
            json.dump(item, f)
            f.write('\n')