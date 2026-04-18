import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("Superstore.csv")

# Create Month column
df["Month"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(range(len(df)), unit="D")
df["Month"] = pd.to_datetime(df["Month"]).dt.month

# Create Unit Price
df["Unit_Price"] = df["Sales"] / df["Quantity"]

# Aggregate monthly demand by product
monthly = df.groupby(["Month","Sub-Category"]).agg({
    "Quantity":"sum",
    "Unit_Price":"mean"
}).reset_index()

# One-hot encode Sub-Category
monthly = pd.get_dummies(monthly, columns=["Sub-Category"])

# Define X and Y
X = monthly.drop("Quantity", axis=1)
y = monthly["Quantity"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, pred))

# Save model
joblib.dump(model, "demand_model.pkl")

print("Model saved successfully")