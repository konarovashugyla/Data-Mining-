import pandas as pd
df = pd.read_csv('your_file.csv')

missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

df_dropped = df.dropna()
print("DataFrame after removing missing values:\n", df_dropped.head())

df_filled_mean = df.fillna(df.mean())
print("DataFrame after filling missing values with mean:\n", df_filled_mean.head())

df_filled_median = df.fillna(df.median())
print("DataFrame after filling missing values with median:\n", df_filled_median.head())

df_filled_value = df.fillna(0)
print("DataFrame after filling missing values with 0:\n", df_filled_value.head())

df_ffill = df.ffill()
print("DataFrame after forward filling missing values:\n", df_ffill.head())

df_bfill = df.bfill()
print("DataFrame after backward filling missing values:\n", df_bfill.head())
print("Original DataFrame:\n", df.head())
print("DataFrame after dropping rows with missing values:\n", df_dropped.head())
print("DataFrame after filling missing values with mean:\n", df_filled_mean.head())
print("DataFrame after filling missing values with median:\n", df_filled_median.head())
print("DataFrame after filling missing values with 0:\n", df_filled_value.head())
print("DataFrame after forward filling:\n", df_ffill.head())
print("DataFrame after backward filling:\n", df_bfill.head())

