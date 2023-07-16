import os
import pandas as pd
import shutil

df = pd.read_csv('data.csv')
files = []
transcritions = []

for i in range(900):
    if pd.isnull(df.at[i, "edited"]):
        transcrition = df.iloc[i]["text"]
    else:
        transcrition = df.iloc[i]["edited"]
    shutil.copy("../trimmed_audios/"+df.at[i, "file"], "../data/")
    files.append(df.at[i, "file"])
    transcritions.append(transcrition)

data = pd.DataFrame({'file': files, 'text': transcritions})
data.to_csv("df.csv")