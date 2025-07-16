import pandas as pd
import os

folder = "/Users/natashajha/Downloads/21CSB0F06_21CSB0A14_SEMINAR/21CSB0F06_21CSB0A14_CODES/embeddings/data"  # Change to your actual folder
files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])

df_list = []
for f in files:
    path = os.path.join(folder, f)
    df = pd.read_csv(path)
    df_list.append(df)

merged = pd.concat(df_list)
merged = merged[["id", "text_embedding", "label"]]  # Only keep relevant columns
merged.to_pickle("openai_master.pkl", protocol=4)

print(f"✅ Saved openai_master.pkl with shape: {merged.shape}")
