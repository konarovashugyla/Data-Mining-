from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv('/mnt/data/Heart_Disease_Prediction.csv')
scaler = MinMaxScaler()
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print(df[numerical_cols].head())
from sklearn.preprocessing import StandardScaler

scaler_z = StandardScaler()
df[numerical_cols] = scaler_z.fit_transform(df[numerical_cols])
print(df[numerical_cols].head())

df_encoded = pd.get_dummies(df, drop_first=True)
print(df_encoded.head())
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False)
categorical_cols = df.select_dtypes(include=['object']).columns
encoded_data = encoder.fit_transform(df[categorical_cols])
df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)
print(df.head())

df['age_binned'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle-aged', 'Old', 'Very Old'])
print(df[['age', 'age_binned']].head())
