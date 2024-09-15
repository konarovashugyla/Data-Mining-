import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_csv('/mnt/data/Heart_Disease_Prediction.csv')

### 1
# Example: Creating polynomial features (degree 2) for 'age' and 'cholesterol'
interaction_features = df[['age', 'cholesterol']]
poly = PolynomialFeatures(degree=2, include_bias=False)
interaction_transformed = poly.fit_transform(interaction_features)
interaction_df = pd.DataFrame(interaction_transformed, columns=poly.get_feature_names_out(['age', 'cholesterol']))
df = pd.concat([df, interaction_df], axis=1)

### 2
# Convert 'date_of_exam' to datetime and extract year, month, day (assuming such a column exists)
df['date_of_exam'] = pd.to_datetime(df['date_of_exam'])
df['exam_year'] = df['date_of_exam'].dt.year
df['exam_month'] = df['date_of_exam'].dt.month
df['exam_day'] = df['date_of_exam'].dt.day

### 3
# Create a new feature based on high cholesterol and age > 50
df['high_risk_age_cholesterol'] = ((df['cholesterol'] > 200) & (df['age'] > 50)).astype(int)
print(df.head())
