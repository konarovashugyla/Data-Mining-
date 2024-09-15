from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

df = pd.read_csv('/mnt/data/Heart_Disease_Prediction.csv')

### 1
interaction_features = df[['age', 'cholesterol']]
poly = PolynomialFeatures(degree=2, include_bias=False)
interaction_transformed = poly.fit_transform(interaction_features)
df = pd.concat([df, pd.DataFrame(interaction_transformed, columns=poly.get_feature_names_out(['age', 'cholesterol']))], axis=1)
### 2
df['date_of_exam'] = pd.to_datetime(df['date_of_exam'])
df['exam_year'] = df['date_of_exam'].dt.year
df['exam_month'] = df['date_of_exam'].dt.month
df['exam_day'] = df['date_of_exam'].dt.day
### 3
df['high_risk_age_cholesterol'] = ((df['cholesterol'] > 200) & (df['age'] > 50)).astype(int)

print(df.head())
