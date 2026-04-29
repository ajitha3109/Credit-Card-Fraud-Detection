import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Load data
df = pd.read_csv("newdata.csv")

# Clean
df = df.dropna(subset=['Class'])
df['Class'] = df['Class'].astype(int)

# Sample (reduce size)
df = df.sample(n=20000, random_state=42)

X = df.drop('Class', axis=1)
y = df['Class']

# 🔥 BALANCE DATA (VERY IMPORTANT)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# Save
pickle.dump(model, open("fraud_model.pkl", "wb"))

print("Model saved successfully!")