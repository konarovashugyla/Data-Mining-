import pandas as pd

# Load the dataset
df = pd.read_csv('/mnt/data/Heart_Disease_Prediction.csv')
### 1
missing_values = df.isnull().sum()
### 2
df_dropped = df.dropna()
### 3
df_filled_mean = df.fillna(df.mean())
df_filled_median = df.fillna(df.median())
df_filled_value = df.fillna(0)  # Example: Filling with 0
### 4
df_ffill = df.ffill()
df_bfill = df.bfill()

print(df.head(), df_dropped.head(), df_filled_mean.head(), df_ffill.head())
