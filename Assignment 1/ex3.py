from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import pandas as pd

df = pd.read_csv('/mnt/data/Heart_Disease_Prediction.csv')

### 1
scaler = MinMaxScaler()  # Use StandardScaler() for Z-score
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

### 2
df_encoded = pd.get_dummies(df, drop_first=True)
encoder = OneHotEncoder(drop='first', sparse=False)
categorical_cols = df.select_dtypes(include=['object']).columns
encoded_data = encoder.fit_transform(df[categorical_cols])
df_encoded_ohe = pd.concat([df.drop(columns=categorical_cols), pd.DataFrame(encoded_data)], axis=1)
### 3
df['age_binned'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle-aged', 'Old', 'Very Old'])

print(df.head())
