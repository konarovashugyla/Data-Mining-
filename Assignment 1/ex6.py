from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('/mnt/data/Heart_Disease_Prediction.csv')
### 1
# Assuming 'target' is the column name for the target variable (replace with actual name)
X = df.drop(columns=['target'])
y = df['target']

### 2

# Example: 80-20 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example: 70-30 Split
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, test_size=0.3, random_state=42)

### 3
print(f"80-20 Split -> Training size: {len(X_train)}, Testing size: {len(X_test)}")
print(f"70-30 Split -> Training size: {len(X_train_70)}, Testing size: {len(X_test_30)}")
