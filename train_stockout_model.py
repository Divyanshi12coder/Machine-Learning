import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/SampleSuperstore.csv")

# create artificial date
df["Order_Date"] = pd.date_range(
    start="2022-01-01",
    periods=len(df),
    freq="D"
)

df["Month"] = df["Order_Date"].dt.month

# unit price
df["Unit_Price"] = df["Sales"] / df["Quantity"]

# simulate inventory level
df["Current_Stock"] = np.random.randint(50,500,len(df))

# stockout logic
df["Stockout"] = df["Quantity"] > df["Current_Stock"]

# FEATURES (match dashboard)
X = df[["Quantity","Current_Stock"]]

y = df["Stockout"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))

joblib.dump(model,"models/stockout_model.pkl")

print("Stockout model saved")