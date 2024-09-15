import pandas as pd
from google.colab import files
uploaded = files.upload()

df = pd.read_csv('data.csv')
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nDataFrame info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nData types of each column:")
print(df.dtypes)
