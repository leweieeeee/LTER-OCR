import json
import cv2
import pymupdf as fitz
import pandas as pd
import numpy as np


pdf_path = "./Incoming data/SBS_DETRITUS_202605.pdf"
json_path = "./output/extracted_res.json"
cell_path = "./output/cell_res.json"

with open (cell_path) as f:
    data = json.load(f)
df = pd.DataFrame(data["boxes"])

df[["x1","y1","x2","y2"]] = df["coordinate"].tolist()

xtolerance = (df["x1"] - df["x2"]).abs().min()/2
ytolerance = (df["y1"] - df["y2"]).abs().min()/2

df = df.drop(columns=["coordinate","cls_id","label"])
df = df.sort_values("x1").reset_index(drop=True)
df["idx"] = (df["x1"].diff() > xtolerance).cumsum()

df = df.sort_values("y1").reset_index(drop=True)
df["idy"] = (df["y1"].diff() > ytolerance).cumsum() 

#print(df.iloc[1:1000].to_string())


xdf = df.groupby("idx").agg({
    "x1": "min",
    "x2": "max",
})

ydf = df.groupby("idy").agg({
    "y1": "min",
    "y2": "max"
})

print(xdf.to_string())
print(ydf.to_string())

with open (json_path, "r", encoding="utf-8") as g:
    data = json.load(g)

ndf = pd.DataFrame({
    "text": data["rec_texts"],
    "poly": data["rec_polys"],
    "conf": data["rec_scores"]
    })

ndf = ndf[ndf["conf"] > 0.90]

ndf[["xpos","ypos"]] = ndf["poly"].apply(lambda p:np.mean(p,axis=0)).tolist()

def find_idx(x):
    matches = xdf.index[
        (xdf["x1"] < x) & (x < xdf["x2"])
    ]
    return (matches[0] if len(matches)>0 else None)

def find_idy(y):
    matches =ydf.index[
        (ydf["y1"] < y) & (y < ydf["y2"])
    ]
    return (matches[0] if len(matches)>0 else None)

ndf["idx"]= ndf["xpos"].apply(find_idx)
ndf["idy"]= ndf["ypos"].apply(find_idy)
ndf = ndf.sort_values("idx").reset_index(drop=True)



print (ndf[["idx","idy","xpos","ypos","text"]].to_string())

result = ndf.pivot_table(
    index = "idx",
    columns = "idy",
    values = "text",
    aggfunc=lambda x:" ".join(x)
).transpose( )

result = result.fillna("")

result.to_excel("output0.9.xlsx")