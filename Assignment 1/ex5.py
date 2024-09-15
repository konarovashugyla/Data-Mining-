import pandas as pd
from scipy import stats

df = pd.read_csv('/mnt/data/Heart_Disease_Prediction.csv')

### 1
df_cleaned = df.drop_duplicates()

### 2
# a
z_scores = stats.zscore(df.select_dtypes(include=['float64', 'int64']))
abs_z_scores = abs(z_scores)
df_no_outliers_z = df_cleaned[(abs_z_scores < 3).all(axis=1)]
# b
Q1 = df_cleaned.quantile(0.25)
Q3 = df_cleaned.quantile(0.75)
IQR = Q3 - Q1
df_no_outliers_iqr = df_cleaned[~((df_cleaned < (Q1 - 1.5 * IQR)) | (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]

### 3
df_cleaned['gender'] = df_cleaned['gender'].str.lower()  # Convert to lowercase
df_cleaned['gender'] = df_cleaned['gender'].replace({'male ': 'male', 'fem': 'female'})  # Correct similar categories

print(df_cleaned.head())
