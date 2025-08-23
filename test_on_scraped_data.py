import pandas as pd
df = pd.read_csv("data/fedex_reviews_20250822_1657.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Sample text: {df['text'].iloc[0]}")