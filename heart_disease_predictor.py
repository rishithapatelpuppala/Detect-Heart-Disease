import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("heart_disease.csv")

df.rename(columns={
    'chest pain type': 'chestpaintype',
    'resting bp s': 'restingbps',
    'fasting blood sugar': 'fastingbloodsugar',
    'resting ecg': 'restingecg',
    'max heart rate': 'maxheartrate',
    'exercise angina': 'exerciseangina',
    'ST slope': 'stslope'
}, inplace=True)

df = pd.get_dummies(df, columns=['chestpaintype', 'restingecg', 'stslope'])

scaler = StandardScaler()
numeric_cols = ['age', 'restingbps', 'cholesterol', 'maxheartrate', 'oldpeak']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
