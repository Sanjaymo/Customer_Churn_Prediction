import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/Telco-Customer-Churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

cat_cols = df.select_dtypes(include=["object","category"]).columns

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Churn", axis=1)
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

model = XGBClassifier(
    n_estimators=450,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.3,
    reg_lambda=2,
    reg_alpha=0.5,
    random_state=42
)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Test Accuracy:", accuracy_score(y_test,y_pred))

print("\nEnter new customer data\n")

user_data = {}

for col in X.columns:
    
    if col in encoders:
        print(f"\nPossible values for {col}: {list(encoders[col].classes_)}")
        val = input(f"Enter value for {col}: ")
        val = encoders[col].transform([val])[0]
    else:
        val = float(input(f"Enter value for {col}: "))
        
    user_data[col] = val

user_df = pd.DataFrame([user_data])

prediction = model.predict(user_df)
prob = model.predict_proba(user_df)

print("\nPrediction:", prediction)

if prediction[0] == 1:
    print("Customer WILL churn")
else:
    print("Customer will NOT churn")

print("Churn Probability:", prob[0][1])