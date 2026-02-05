import os
import pandas as pd

SONGS_DIR = "songs"

songs_data = []

for filename in os.listdir(SONGS_DIR):
    file_path = os.path.join(SONGS_DIR, filename)
    
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        songs_data.append({
            "title": filename,  
            "artist": "ATL",
            "text": text
        })

df = pd.DataFrame(songs_data)

df.to_csv("songs.csv", index=False, encoding="utf-8-sig")

print(f"{len(df)} песен сохранены в songs.csv")
