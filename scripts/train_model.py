import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# make sure model folder exists
os.makedirs("model", exist_ok=True)

# load dataset
data = pd.read_csv("data/returns.csv")

# separate features and label
X = data.drop("fraud_label", axis=1)
y = data["fraud_label"]

# create model
model = RandomForestClassifier(n_estimators=120)

# train model
model.fit(X, y)

# save model
joblib.dump(model, "model/fraud_model.pkl")

print("Model trained and saved successfully!")