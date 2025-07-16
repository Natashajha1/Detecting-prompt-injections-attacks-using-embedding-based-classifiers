# generate_test_split.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("dataset", exist_ok=True)

df = pd.read_pickle("openai_master.pkl")
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

test_df[["id"]].to_csv("dataset/test_indices.csv", index=False)
print("✅ Created 'dataset/test_indices.csv'")
